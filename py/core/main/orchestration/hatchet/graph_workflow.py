import asyncio
import json
import logging
import math
import time
import uuid
from datetime import timedelta
from typing import Any

from hatchet_sdk import (
    ConcurrencyExpression,
    ConcurrencyLimitStrategy,
    Context,
    Hatchet,
)
from pydantic import BaseModel

from core import GenerationConfig
from core.base import OrchestrationProvider, R2RException
from core.base.abstractions import (
    GraphConstructionStatus,
    GraphExtractionStatus,
)

from ...services import GraphService

logger = logging.getLogger()


def hatchet_graph_search_results_factory(
    hatchet: Hatchet,
    orchestration_provider: OrchestrationProvider,
    service: GraphService,
) -> dict[str, "Hatchet.Workflow"]:
    def convert_to_dict(input_data):
        """Converts input data back to a plain dictionary format, handling
        special cases like UUID and GenerationConfig. This is the inverse of
        get_input_data_dict.

        Args:
            input_data: Dictionary containing the input data with potentially special types

        Returns:
            Dictionary with all values converted to basic Python types
        """
        output_data = {}

        for key, value in input_data.items():
            if value is None:
                output_data[key] = None
                continue

            # Convert UUID to string
            if isinstance(value, uuid.UUID):
                output_data[key] = str(value)

            try:
                output_data[key] = value.model_dump()
            except Exception:
                # Handle nested dictionaries that might contain settings
                if isinstance(value, dict):
                    output_data[key] = convert_to_dict(value)

                # Handle lists that might contain dictionaries
                elif isinstance(value, list):
                    output_data[key] = [
                        (
                            convert_to_dict(item)
                            if isinstance(item, dict)
                            else item
                        )
                        for item in value
                    ]

                # All other types can be directly assigned
                else:
                    output_data[key] = value

        return output_data

    def get_input_data_dict(input_data):
        for key, value in input_data.items():
            if value is None:
                continue

            if key == "document_id":
                input_data[key] = (
                    value if isinstance(value, uuid.UUID) else uuid.UUID(value)
                )

            if key == "collection_id":
                input_data[key] = (
                    value if isinstance(value, uuid.UUID) else uuid.UUID(value)
                )

            if key == "graph_id":
                input_data[key] = (
                    value if isinstance(value, uuid.UUID) else uuid.UUID(value)
                )

            if key in ["graph_creation_settings", "graph_enrichment_settings"]:
                # Ensure we have a dict (if not already)
                input_data[key] = (
                    value if isinstance(value, dict) else json.loads(value)
                )

                if "generation_config" in input_data[key]:
                    gen_cfg = input_data[key]["generation_config"]
                    # If it's a dict, convert it
                    if isinstance(gen_cfg, dict):
                        input_data[key]["generation_config"] = (
                            GenerationConfig(**gen_cfg)
                        )
                    # If it's not already a GenerationConfig, default it
                    elif not isinstance(gen_cfg, GenerationConfig):
                        input_data[key]["generation_config"] = (
                            GenerationConfig()
                        )

                    input_data[key]["generation_config"].model = (
                        input_data[key]["generation_config"].model
                        or service.config.app.fast_llm
                    )

        return input_data

    class GraphExtractionInput(BaseModel):
        example: str

    class GraphClusteringInput(BaseModel):
        example: str

    class GraphCommunitySummarizationInput(BaseModel):
        example: str

    class GraphDeduplicationInput(BaseModel):
        example: str

    graph_extraction_workflow = hatchet.workflow(
        name="graph-extraction",
        input_validator=GraphExtractionInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @graph_extraction_workflow.task(execution_timeout=timedelta(minutes=360))
    async def graph_extraction(
        input: GraphExtractionInput, ctx: Context
    ) -> dict[str, Any]:
        request = ctx.workflow_input()["request"]

        input_data = get_input_data_dict(request)
        document_id = input_data.get("document_id", None)
        collection_id = input_data.get("collection_id", None)

        await service.providers.database.documents_handler.set_workflow_status(
            id=document_id,
            status_type="extraction_status",
            status=GraphExtractionStatus.PROCESSING,
        )

        if collection_id and not document_id:
            document_ids = await service.get_document_ids_for_create_graph(
                collection_id=collection_id,
                **input_data["graph_creation_settings"],
            )
            workflows = []

            for document_id in document_ids:
                input_data_copy = input_data.copy()
                input_data_copy["collection_id"] = str(
                    input_data_copy["collection_id"]
                )
                input_data_copy["document_id"] = str(document_id)

                workflows.append(
                    ctx.aio.spawn_workflow(
                        "graph-extraction",
                        {
                            "request": {
                                **convert_to_dict(input_data_copy),
                            }
                        },
                        key=str(document_id),
                    )
                )
            # Wait for all workflows to complete
            results = await asyncio.gather(*workflows)
            return {
                "result": f"successfully submitted graph_search_results relationships extraction for document {document_id}",
                "document_id": str(collection_id),
            }

        else:
            # Extract relationships and store them
            extractions = []
            async for extraction in service.graph_search_results_extraction(
                document_id=document_id,
                **input_data["graph_creation_settings"],
            ):
                logger.info(
                    f"Found extraction with {len(extraction.entities)} entities"
                )
                extractions.append(extraction)

            await service.store_graph_search_results_extractions(extractions)

            logger.info(
                f"Successfully ran graph_search_results relationships extraction for document {document_id}"
            )

            return {
                "result": f"successfully ran graph_search_results relationships extraction for document {document_id}",
                "document_id": str(document_id),
            }

    @graph_extraction_workflow.task(
        parents=[graph_extraction], execution_timeout=timedelta(minutes=360)
    )
    async def graph_extraction_entity_description(
        ctx: Context,
    ) -> dict[str, Any]:
        input_data = get_input_data_dict(ctx.workflow_input()["request"])
        document_id = input_data.get("document_id", None)

        # Describe the entities in the graph
        await service.graph_search_results_entity_description(
            document_id=document_id,
            **input_data["graph_creation_settings"],
        )

        logger.info(
            f"Successfully ran graph_search_results entity description for document {document_id}"
        )

        if service.providers.database.config.graph_creation_settings.automatic_deduplication:
            extract_input = {
                "document_id": str(document_id),
            }

            extract_result = (
                await ctx.aio.spawn_workflow(
                    "graph-deduplication",
                    {"request": extract_input},
                )
            ).result()

            await asyncio.gather(extract_result)

        return {
            "result": f"successfully ran graph_search_results entity description for document {document_id}"
        }

    @graph_extraction_workflow.on_failure_task()
    async def graph_extraction_on_failure(ctx: Context) -> None:
        input_data = get_input_data_dict(ctx.workflow_input()["request"])
        document_id = input_data.get("document_id", None)

        if not document_id:
            logger.info(
                "No document id was found in workflow input to mark a failure."
            )
            return

        try:
            await service.providers.database.documents_handler.set_workflow_status(
                id=uuid.UUID(document_id),
                status_type="extraction_status",
                status=GraphExtractionStatus.FAILED,
            )
            logger.info(
                f"Updated Graph extraction status for {document_id} to FAILED"
            )
        except Exception as e:
            logger.error(
                f"Failed to update document status for {document_id}: {e}"
            )

    graph_clustering_workflow = hatchet.workflow(
        name="graph-clustering",
        input_validator=GraphClusteringInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @graph_clustering_workflow.task(execution_timeout=timedelta(minutes=360))
    async def graph_search_results_clustering(
        input: GraphClusteringInput, ctx: Context
    ) -> dict[str, Any]:
        logger.info("Running Graph Clustering")

        input_data = get_input_data_dict(ctx.workflow_input()["request"])

        # Get the collection_id and graph_id
        collection_id = input_data.get("collection_id", None)
        graph_id = input_data.get("graph_id", None)

        # Check current workflow status
        workflow_status = await service.providers.database.documents_handler.get_workflow_status(
            id=collection_id,
            status_type="graph_cluster_status",
        )

        if workflow_status == GraphConstructionStatus.SUCCESS:
            raise R2RException(
                "Communities have already been built for this collection. To build communities again, first reset the graph.",
                400,
            )

        # Run clustering
        try:
            graph_search_results_clustering_results = (
                await service.graph_search_results_clustering(
                    collection_id=collection_id,
                    graph_id=graph_id,
                    **input_data["graph_enrichment_settings"],
                )
            )

            num_communities = graph_search_results_clustering_results[
                "num_communities"
            ][0]

            if num_communities == 0:
                raise R2RException("No communities found", 400)

            return {
                "result": graph_search_results_clustering_results,
            }
        except Exception as e:
            await service.providers.database.documents_handler.set_workflow_status(
                id=collection_id,
                status_type="graph_cluster_status",
                status=GraphConstructionStatus.FAILED,
            )
            raise e

    @graph_clustering_workflow.task(
        parents=[graph_search_results_clustering],
        execution_timeout=timedelta(minutes=360),
    )
    async def graph_search_results_community_summary_clustering(
        ctx: Context,
    ) -> dict[str, Any]:
        input_data = get_input_data_dict(ctx.workflow_input()["request"])
        collection_id = input_data.get("collection_id", None)
        graph_id = input_data.get("graph_id", None)
        # Get number of communities from previous step
        num_communities = ctx.step_output("graph_search_results_clustering")[
            "result"
        ]["num_communities"][0]

        # Calculate batching
        parallel_communities = min(100, num_communities)
        total_workflows = math.ceil(num_communities / parallel_communities)
        workflows = []

        logger.info(
            f"Running Graph Community Summary for {num_communities} communities, spawning {total_workflows} workflows"
        )

        # Spawn summary workflows
        for i in range(total_workflows):
            offset = i * parallel_communities
            limit = min(parallel_communities, num_communities - offset)

            workflows.append(
                (
                    await ctx.aio.spawn_workflow(
                        "graph-community-summarization",
                        {
                            "request": {
                                "offset": offset,
                                "limit": limit,
                                "graph_id": (
                                    str(graph_id) if graph_id else None
                                ),
                                "collection_id": (
                                    str(collection_id)
                                    if collection_id
                                    else None
                                ),
                                "graph_enrichment_settings": convert_to_dict(
                                    input_data["graph_enrichment_settings"]
                                ),
                            }
                        },
                        key=f"{i}/{total_workflows}_community_summary",
                    )
                ).result()
            )

        results = await asyncio.gather(*workflows)
        logger.info(f"Completed {len(results)} community summary workflows")

        # Update statuses
        document_ids = await service.providers.database.documents_handler.get_document_ids_by_status(
            status_type="extraction_status",
            status=GraphExtractionStatus.SUCCESS,
            collection_id=collection_id,
        )

        await service.providers.database.documents_handler.set_workflow_status(
            id=document_ids,
            status_type="extraction_status",
            status=GraphExtractionStatus.ENRICHED,
        )

        await service.providers.database.documents_handler.set_workflow_status(
            id=collection_id,
            status_type="graph_cluster_status",
            status=GraphConstructionStatus.SUCCESS,
        )

        return {
            "result": f"Successfully completed enrichment with {len(results)} summary workflows"
        }

    @graph_clustering_workflow.on_failure_task()
    async def graph_clustering_on_failure(ctx: Context) -> None:
        collection_id = ctx.workflow_input()["request"].get(
            "collection_id", None
        )
        if collection_id:
            await service.providers.database.documents_handler.set_workflow_status(
                id=uuid.UUID(collection_id),
                status_type="graph_cluster_status",
                status=GraphConstructionStatus.FAILED,
            )

    graph_community_summarization_workflow = hatchet.workflow(
        name="graph-community-summarization",
        input_validator=GraphCommunitySummarizationInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @graph_community_summarization_workflow.task(
        execution_timeout=timedelta(minutes=360)
    )
    async def graph_search_results_community_summary(
        input: GraphCommunitySummarizationInput, ctx: Context
    ) -> dict[str, Any]:
        start_time = time.time()

        input_data = get_input_data_dict(ctx.workflow_input()["request"])

        base_args = {
            k: v
            for k, v in input_data.items()
            if k != "graph_enrichment_settings"
        }
        enrichment_args = input_data.get("graph_enrichment_settings", {})

        # Merge them together.
        # Note: if there is any key overlap, values from enrichment_args will override those from base_args.
        merged_args = {**base_args, **enrichment_args}

        # Now call the service method with all arguments at the top level.
        # This ensures that keys like "max_summary_input_length" and "generation_config" are present.
        community_summary = (
            await service.graph_search_results_community_summary(**merged_args)
        )
        logger.info(
            f"Successfully ran graph_search_results community summary for communities {input_data['offset']} to {input_data['offset'] + len(community_summary)} in {time.time() - start_time:.2f} seconds "
        )
        return {
            "result": f"successfully ran graph_search_results community summary for communities {input_data['offset']} to {input_data['offset'] + len(community_summary)}"
        }

    graph_deduplication_workflow = hatchet.workflow(
        name="graph-deduplication",
        input_validator=GraphDeduplicationInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @graph_deduplication_workflow.task(
        execution_timeout=timedelta(minutes=360)
    )
    async def graph_deduplication(
        input: GraphDeduplicationInput, ctx: Context
    ) -> dict[str, Any]:
        start_time = time.time()

        input_data = get_input_data_dict(ctx.workflow_input()["request"])

        document_id = input_data.get("document_id", None)

        await service.deduplicate_document_entities(
            document_id=document_id,
        )
        logger.info(
            f"Successfully ran deduplication for document {document_id} in {time.time() - start_time:.2f} seconds "
        )
        return {
            "result": f"Successfully ran deduplication for document {document_id}"
        }

    return {
        "graph-extraction": graph_extraction_workflow,
        "graph-clustering": graph_clustering_workflow,
        "graph-community-summarization": graph_community_summarization_workflow,
        "graph-deduplication": graph_deduplication_workflow,
    }
