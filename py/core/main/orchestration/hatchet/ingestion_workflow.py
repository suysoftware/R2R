import asyncio
import logging
from datetime import timedelta
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from hatchet_sdk import (
    ConcurrencyExpression,
    ConcurrencyLimitStrategy,
    Context,
    Hatchet,
)
from litellm import AuthenticationError
from pydantic import BaseModel

from core.base import (
    DocumentChunk,
    GraphConstructionStatus,
    IngestionStatus,
    OrchestrationProvider,
    generate_extraction_id,
)
from core.base.abstractions import DocumentResponse, R2RException
from core.utils import (
    generate_default_user_collection_id,
    num_tokens,
    update_settings_from_dict,
)

from ...services import IngestionService, IngestionServiceAdapter

logger = logging.getLogger()


def hatchet_ingestion_factory(
    hatchet: Hatchet,
    orchestration_provider: OrchestrationProvider,
    service: IngestionService,
) -> dict[str, "Hatchet.workflow"]:
    class IngestFilesInput(BaseModel):
        example: str

    class IngestChuksInput(BaseModel):
        example: str

    class UpdateChunkInput(BaseModel):
        example: str

    class CreateVectorIndexInput(BaseModel):
        example: str

    class DeleteVectorIndexInput(BaseModel):
        example: str

    ingest_files_workflow = hatchet.workflow(
        name="ingest-files",
        input_validator=IngestFilesInput,
        concurrency=ConcurrencyExpression(
            expression="input.example",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    logger.warning(
        f"ingest_files_workflow: {ingest_files_workflow} is of type {type(ingest_files_workflow)}"
    )

    @ingest_files_workflow.task(execution_timeout=timedelta(minutes=5))
    async def parse(input: IngestFilesInput, ctx: Context) -> dict[str, Any]:
        try:
            logger.info("Initiating ingestion workflow, step: parse")
            logger.warning(f"input: {input}\n\nctx: {ctx}")
            input_data = ctx.workflow_input()["request"]
            parsed_data = IngestionServiceAdapter.parse_ingest_file_input(
                input_data
            )

            document_info = service.create_document_info_from_file(
                parsed_data["document_id"],
                parsed_data["user"],
                parsed_data["file_data"]["filename"],
                parsed_data["metadata"],
                parsed_data["version"],
                parsed_data["size_in_bytes"],
            )

            await service.update_document_status(
                document_info,
                status=IngestionStatus.PARSING,
            )

            ingestion_config = parsed_data["ingestion_config"] or {}
            extractions_generator = service.parse_file(
                document_info, ingestion_config
            )

            extractions = []
            async for extraction in extractions_generator:
                extractions.append(extraction)

            # 2) Sum tokens
            total_tokens = 0
            for chunk in extractions:
                text_data = chunk.data
                if not isinstance(text_data, str):
                    text_data = text_data.decode("utf-8", errors="ignore")
                total_tokens += num_tokens(text_data)
            document_info.total_tokens = total_tokens

            if not ingestion_config.get("skip_document_summary", False):
                await service.update_document_status(
                    document_info, status=IngestionStatus.AUGMENTING
                )
                await service.augment_document_info(
                    document_info,
                    [extraction.to_dict() for extraction in extractions],
                )

            await service.update_document_status(
                document_info,
                status=IngestionStatus.EMBEDDING,
            )

            embedding_generator = service.embed_document(
                [extraction.to_dict() for extraction in extractions]
            )

            embeddings = []
            async for embedding in embedding_generator:
                embeddings.append(embedding)

            await service.update_document_status(
                document_info,
                status=IngestionStatus.STORING,
            )

            storage_generator = service.store_embeddings(  # type: ignore
                embeddings
            )

            async for _ in storage_generator:
                pass

            await service.finalize_ingestion(document_info)

            await service.update_document_status(
                document_info,
                status=IngestionStatus.SUCCESS,
            )

            collection_ids = ctx.workflow_input()["request"].get(
                "collection_ids"
            )
            if not collection_ids:
                # TODO: Move logic onto the `management service`
                collection_id = generate_default_user_collection_id(
                    document_info.owner_id
                )
                await service.providers.database.collections_handler.assign_document_to_collection_relational(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )
                await service.providers.database.chunks_handler.assign_document_chunks_to_collection(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.OUTDATED,
                )
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_cluster_status",  # NOTE - we should actually check that cluster has been made first, if not it should be PENDING still
                    status=GraphConstructionStatus.OUTDATED,
                )
            else:
                for collection_id_str in collection_ids:
                    collection_id = UUID(collection_id_str)
                    try:
                        name = document_info.title or "N/A"
                        description = ""
                        await service.providers.database.collections_handler.create_collection(
                            owner_id=document_info.owner_id,
                            name=name,
                            description=description,
                            collection_id=collection_id,
                        )
                        await service.providers.database.graphs_handler.create(
                            collection_id=collection_id,
                            name=name,
                            description=description,
                            graph_id=collection_id,
                        )

                    except Exception as e:
                        logger.warning(
                            f"Warning, could not create collection with error: {str(e)}"
                        )

                    await service.providers.database.collections_handler.assign_document_to_collection_relational(
                        document_id=document_info.id,
                        collection_id=collection_id,
                    )
                    await service.providers.database.chunks_handler.assign_document_chunks_to_collection(
                        document_id=document_info.id,
                        collection_id=collection_id,
                    )
                    await service.providers.database.documents_handler.set_workflow_status(
                        id=collection_id,
                        status_type="graph_sync_status",
                        status=GraphConstructionStatus.OUTDATED,
                    )
                    await service.providers.database.documents_handler.set_workflow_status(
                        id=collection_id,
                        status_type="graph_cluster_status",  # NOTE - we should actually check that cluster has been made first, if not it should be PENDING still
                        status=GraphConstructionStatus.OUTDATED,
                    )

            # get server chunk enrichment settings and override parts of it if provided in the ingestion config
            if server_chunk_enrichment_settings := getattr(
                service.providers.ingestion.config,
                "chunk_enrichment_settings",
                None,
            ):
                chunk_enrichment_settings = update_settings_from_dict(
                    server_chunk_enrichment_settings,
                    ingestion_config.get("chunk_enrichment_settings", {})
                    or {},
                )

            if chunk_enrichment_settings.enable_chunk_enrichment:
                logger.info("Enriching document with contextual chunks")

                document_info: DocumentResponse = (
                    await service.providers.database.documents_handler.get_documents_overview(
                        offset=0,
                        limit=1,
                        filter_user_ids=[document_info.owner_id],
                        filter_document_ids=[document_info.id],
                    )
                )["results"][0]

                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.ENRICHING,
                )

                await service.chunk_enrichment(
                    document_id=document_info.id,
                    document_summary=document_info.summary,
                    chunk_enrichment_settings=chunk_enrichment_settings,
                )

                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.SUCCESS,
                )

            if service.providers.ingestion.config.automatic_extraction:
                extract_input = {
                    "document_id": str(document_info.id),
                    "graph_creation_settings": service.providers.database.config.graph_creation_settings.model_dump_json(),
                    "user": input_data["user"],
                }

                extract_result = (
                    await ctx.aio.spawn_workflow(
                        "graph-extraction",
                        {"request": extract_input},
                    )
                ).result()

                await asyncio.gather(extract_result)

            return {
                "status": "Successfully finalized ingestion",
                "document_info": document_info.to_dict(),
            }

        except AuthenticationError:
            raise R2RException(
                status_code=401,
                message="Authentication error: Invalid API key or credentials.",
            ) from None
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during ingestion: {str(e)}",
            ) from e

    @ingest_files_workflow.on_failure_task()
    async def ingest_files_on_failure(
        input: IngestFilesInput, ctx: Context
    ) -> None:
        request = ctx.workflow_input().get("request", {})
        document_id = request.get("document_id")

        if not document_id:
            logger.error(
                "No document id was found in workflow input to mark a failure."
            )
            return

        try:
            documents_overview = (
                await service.providers.database.documents_handler.get_documents_overview(
                    offset=0,
                    limit=1,
                    filter_document_ids=[document_id],
                )
            )["results"]

            if not documents_overview:
                logger.error(
                    f"Document with id {document_id} not found in database to mark failure."
                )
                return

            document_info = documents_overview[0]

            # Update the document status to FAILED
            if document_info.ingestion_status != IngestionStatus.SUCCESS:
                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.FAILED,
                    metadata={"failure": f"{ctx.task_run_errors()}"},
                )

        except Exception as e:
            logger.error(
                f"Failed to update document status for {document_id}: {e}"
            )

    ingest_chunks_workflow = hatchet.workflow(
        name="ingest-chunks",
        input_validator=IngestChuksInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @ingest_chunks_workflow.task(
        execution_timeout=timedelta(minutes=60)
    )  # 60 minutes for chunks seems unreasonably high
    async def ingest(input: IngestChuksInput, ctx: Context) -> dict[str, Any]:
        input_data = ctx.workflow_input()["request"]
        parsed_data = IngestionServiceAdapter.parse_ingest_chunks_input(
            input_data
        )

        document_info = await service.ingest_chunks_ingress(**parsed_data)

        await service.update_document_status(
            document_info, status=IngestionStatus.EMBEDDING
        )
        document_id = document_info.id

        extractions = [
            DocumentChunk(
                id=generate_extraction_id(document_id, i),
                document_id=document_id,
                collection_ids=[],
                owner_id=document_info.owner_id,
                data=chunk.text,
                metadata=parsed_data["metadata"],
            ).to_dict()
            for i, chunk in enumerate(parsed_data["chunks"])
        ]

        # 2) Sum tokens
        total_tokens = 0
        for chunk in extractions:
            text_data = chunk["data"]
            if not isinstance(text_data, str):
                text_data = text_data.decode("utf-8", errors="ignore")
            total_tokens += num_tokens(text_data)
        document_info.total_tokens = total_tokens

        return {
            "status": "Successfully ingested chunks",
            "extractions": extractions,
            "document_info": document_info.to_dict(),
        }

    @ingest_chunks_workflow.task(
        parents=[ingest], execution_timeout=timedelta(minutes=60)
    )
    async def embed(input: IngestChuksInput, ctx: Context) -> dict[str, Any]:
        document_info_dict = ctx.task_output("ingest")["document_info"]
        document_info = DocumentResponse(**document_info_dict)

        extractions = ctx.task_output("ingest")["extractions"]

        embedding_generator = service.embed_document(extractions)
        embeddings = [
            embedding.model_dump() async for embedding in embedding_generator
        ]

        await service.update_document_status(
            document_info, status=IngestionStatus.STORING
        )

        storage_generator = service.store_embeddings(embeddings)
        async for _ in storage_generator:
            pass

        return {
            "status": "Successfully embedded and stored chunks",
            "document_info": document_info.to_dict(),
        }

    @ingest_chunks_workflow.task(
        parents=[embed], execution_timeout=timedelta(minutes=60)
    )
    async def finalize(
        input: IngestChuksInput, ctx: Context
    ) -> dict[str, Any]:
        document_info_dict = ctx.task_output("embed")["document_info"]
        document_info = DocumentResponse(**document_info_dict)

        await service.finalize_ingestion(document_info)

        await service.update_document_status(
            document_info, status=IngestionStatus.SUCCESS
        )

        try:
            # TODO - Move logic onto the `management service`
            collection_ids = ctx.workflow_input()["request"].get(
                "collection_ids"
            )
            if not collection_ids:
                # TODO: Move logic onto the `management service`
                collection_id = generate_default_user_collection_id(
                    document_info.owner_id
                )
                await service.providers.database.collections_handler.assign_document_to_collection_relational(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )
                await service.providers.database.chunks_handler.assign_document_chunks_to_collection(
                    document_id=document_info.id,
                    collection_id=collection_id,
                )
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_sync_status",
                    status=GraphConstructionStatus.OUTDATED,
                )
                await service.providers.database.documents_handler.set_workflow_status(
                    id=collection_id,
                    status_type="graph_cluster_status",  # NOTE - we should actually check that cluster has been made first, if not it should be PENDING still
                    status=GraphConstructionStatus.OUTDATED,
                )
            else:
                for collection_id_str in collection_ids:
                    collection_id = UUID(collection_id_str)
                    try:
                        name = document_info.title or "N/A"
                        description = ""
                        await service.providers.database.collections_handler.create_collection(
                            owner_id=document_info.owner_id,
                            name=name,
                            description=description,
                            collection_id=collection_id,
                        )
                        await service.providers.database.graphs_handler.create(
                            collection_id=collection_id,
                            name=name,
                            description=description,
                            graph_id=collection_id,
                        )

                    except Exception as e:
                        logger.warning(
                            f"Warning, could not create collection with error: {str(e)}"
                        )

                    await service.providers.database.collections_handler.assign_document_to_collection_relational(
                        document_id=document_info.id,
                        collection_id=collection_id,
                    )

                    await service.providers.database.chunks_handler.assign_document_chunks_to_collection(
                        document_id=document_info.id,
                        collection_id=collection_id,
                    )

                    await service.providers.database.documents_handler.set_workflow_status(
                        id=collection_id,
                        status_type="graph_sync_status",
                        status=GraphConstructionStatus.OUTDATED,
                    )

                    await service.providers.database.documents_handler.set_workflow_status(
                        id=collection_id,
                        status_type="graph_cluster_status",
                        status=GraphConstructionStatus.OUTDATED,  # NOTE - we should actually check that cluster has been made first, if not it should be PENDING still
                    )
        except Exception as e:
            logger.error(
                f"Error during assigning document to collection: {str(e)}"
            )

        return {
            "status": "Successfully finalized ingestion",
            "document_info": document_info.to_dict(),
        }

    @ingest_chunks_workflow.on_failure_task()
    async def ingest_chunks_on_failure(
        input: IngestChuksInput, ctx: Context
    ) -> None:
        request = ctx.workflow_input().get("request", {})
        document_id = request.get("document_id")

        if not document_id:
            logger.error(
                "No document id was found in workflow input to mark a failure."
            )
            return

        try:
            documents_overview = (
                await service.providers.database.documents_handler.get_documents_overview(
                    offset=0,
                    limit=1,
                    filter_document_ids=[document_id],
                )
            )["results"]

            if not documents_overview:
                logger.error(
                    f"Document with id {document_id} not found in database to mark failure."
                )
                return

            document_info = documents_overview[0]

            # Update the document status to FAILED
            if document_info.ingestion_status != IngestionStatus.SUCCESS:
                await service.update_document_status(
                    document_info,
                    status=IngestionStatus.FAILED,
                    metadata={"failure": f"{ctx.task_run_errors()}"},
                )

        except Exception as e:
            logger.error(
                f"Failed to update document status for {document_id}: {e}"
            )

    update_chunks_workflow = hatchet.workflow(
        name="update-chunks",
        input_validator=UpdateChunkInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @update_chunks_workflow.task(
        execution_timeout=timedelta(minutes=60)
    )  # 60 minutes for chunks seems unreasonably high
    async def update_chunks(
        input: UpdateChunkInput, ctx: Context
    ) -> dict[str, Any]:
        try:
            input_data = ctx.workflow_input()["request"]
            parsed_data = IngestionServiceAdapter.parse_update_chunk_input(
                input_data
            )

            document_uuid = (
                UUID(parsed_data["document_id"])
                if isinstance(parsed_data["document_id"], str)
                else parsed_data["document_id"]
            )
            extraction_uuid = (
                UUID(parsed_data["id"])
                if isinstance(parsed_data["id"], str)
                else parsed_data["id"]
            )

            await service.update_chunk_ingress(
                document_id=document_uuid,
                chunk_id=extraction_uuid,
                text=parsed_data.get("text"),
                user=parsed_data["user"],
                metadata=parsed_data.get("metadata"),
                collection_ids=parsed_data.get("collection_ids"),
            )

            return {
                "message": "Chunk update completed successfully.",
                "task_id": ctx.workflow_run_id(),
                "document_ids": [str(document_uuid)],
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during chunk update: {str(e)}",
            ) from e

    @update_chunks_workflow.on_failure_task()
    async def update_chunks_on_failure(
        input: UpdateChunkInput, ctx: Context
    ) -> None:
        pass

    create_vector_index_workflow = hatchet.workflow(
        name="create-vector-index",
        input_validator=CreateVectorIndexInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @create_vector_index_workflow.task(execution_timeout=timedelta(minutes=60))
    async def create_vector_index(
        input: CreateVectorIndexInput, ctx: Context
    ) -> dict[str, Any]:
        input_data = ctx.workflow_input()["request"]
        parsed_data = IngestionServiceAdapter.parse_create_vector_index_input(
            input_data
        )

        await service.providers.database.chunks_handler.create_index(
            **parsed_data
        )

        return {
            "status": "Vector index creation queued successfully.",
        }

    @create_vector_index_workflow.on_failure_task()
    async def create_vector_index_on_failure(
        input: CreateVectorIndexInput, ctx: Context
    ) -> None:
        pass

    delete_vector_index_workflow = hatchet.workflow(
        name="delete-vector-index",
        input_validator=DeleteVectorIndexInput,
        concurrency=ConcurrencyExpression(
            expression="input.user.id",
            max_runs=1,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
        ),
    )

    @delete_vector_index_workflow.task(execution_timeout=timedelta(minutes=10))
    async def delete_vector_index(
        input: DeleteVectorIndexInput, ctx: Context
    ) -> dict[str, Any]:
        input_data = ctx.workflow_input()["request"]
        parsed_data = IngestionServiceAdapter.parse_delete_vector_index_input(
            input_data
        )

        await service.providers.database.chunks_handler.delete_index(
            **parsed_data
        )

        return {"status": "Vector index deleted successfully."}

    return {
        "ingest_files": ingest_files_workflow,
        "ingest_chunks": ingest_chunks_workflow,
        "update_chunk": update_chunks_workflow,
        "create_vector_index": create_vector_index_workflow,
        "delete_vector_index": delete_vector_index_workflow,
    }
