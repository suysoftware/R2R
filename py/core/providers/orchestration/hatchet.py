# FIXME: Once the Hatchet workflows are type annotated, remove the type: ignore comments
import asyncio
import logging
from typing import Any, Callable, Optional

from core.base import OrchestrationConfig, OrchestrationProvider, Workflow

logger = logging.getLogger()


class HatchetOrchestrationProvider(OrchestrationProvider):
    def __init__(self, config: OrchestrationConfig):
        super().__init__(config)
        try:
            from hatchet_sdk import ClientConfig, Hatchet
        except ImportError:
            raise ImportError(
                "Hatchet SDK not installed. Please install it using `pip install hatchet-sdk`."
            ) from None
        root_logger = logging.getLogger()

        self.hatchet = Hatchet(
            config=ClientConfig(
                logger=root_logger,
            ),
        )
        self.root_logger = root_logger
        self.config: OrchestrationConfig = config
        self.messages: dict[str, str] = {}

    def get_worker(self, name: str, max_runs: Optional[int] = None) -> Any:
        if not max_runs:
            max_runs = self.config.max_runs
        self.worker = self.hatchet.worker(name, max_runs)  # type: ignore
        return self.worker

    def concurrency(self, *args, **kwargs) -> Callable:
        return self.hatchet.concurrency(*args, **kwargs)

    async def start_worker(self):
        if not self.worker:
            raise ValueError(
                "Worker not initialized. Call get_worker() first."
            )

        asyncio.create_task(self.worker.async_start())

    async def run_workflow(
        self,
        workflow_name: str,
        parameters: dict,
        options: dict,
        *args,
        **kwargs,
    ) -> Any:
        task_id = self.hatchet.admin.run_workflow(
            workflow_name,
            parameters,
            options=options,  # type: ignore
            *args,
            **kwargs,
        )
        return {
            "task_id": str(task_id),
            "message": self.messages.get(
                workflow_name, "Workflow queued successfully."
            ),  # Return message based on workflow name
        }

    def register_workflows(
        self, workflow: Workflow, service: Any, messages: dict
    ) -> None:
        self.messages.update(messages)

        logger.info(
            f"Registering workflows for {workflow} with messages {messages}."
        )
        if workflow == Workflow.INGESTION:
            from core.main.orchestration.hatchet.ingestion_workflow import (  # type: ignore
                hatchet_ingestion_factory,
            )

            logger.warning(
                f"Hatchet is {self.hatchet} of type {type(self.hatchet)}"
            )

            workflows = hatchet_ingestion_factory(
                hatchet=self.hatchet,
                orchestration_provider=self,
                service=service,
            )
            if self.worker:
                for workflow in workflows.values():
                    self.worker.register_workflow(workflow)

        elif workflow == Workflow.GRAPH:
            from core.main.orchestration.hatchet.graph_workflow import (  # type: ignore
                hatchet_graph_search_results_factory,
            )

            workflows = hatchet_graph_search_results_factory(
                hatchet=self.hatchet,
                orchestration_provider=self,
                service=service,
            )
            if self.worker:
                for workflow in workflows.values():
                    self.worker.register_workflow(workflow)
