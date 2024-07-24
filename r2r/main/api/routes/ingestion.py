import uuid
from typing import Optional

from fastapi import Depends, File, UploadFile
from pydantic import BaseModel

from ...engine import R2REngine
from ...services.ingestion_service import IngestionService
from .base_router import BaseRouter


class R2RIngestFilesRequest(BaseModel):
    document_ids: Optional[list[uuid.UUID]] = None
    metadatas: Optional[list[dict]] = None
    versions: Optional[list[str]] = None


class R2RUpdateFilesRequest(BaseModel):
    metadatas: Optional[list[dict]] = None
    document_ids: Optional[list[uuid.UUID]] = None


class IngestionRouter(BaseRouter):
    def __init__(self, engine: R2REngine):
        super().__init__(engine)
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/ingest_files")
        @self.base_endpoint
        async def ingest_files_app(
            files: list[UploadFile] = File(...),
            request: R2RIngestFilesRequest = Depends(
                IngestionService.parse_ingest_files_form_data
            ),
            auth_user=Depends(self.engine.providers.auth.auth_wrapper),
        ):
            return await self.engine.aingest_files(
                files=files,
                metadatas=request.metadatas,
                document_ids=request.document_ids,
                versions=request.versions,
                user=auth_user,
            )

        @self.router.post("/update_files")
        @self.base_endpoint
        async def update_files_app(
            files: list[UploadFile] = File(...),
            request: R2RUpdateFilesRequest = Depends(
                IngestionService.parse_update_files_form_data
            ),
            auth_user=Depends(self.engine.providers.auth.auth_wrapper),
        ):
            return await self.engine.aupdate_files(
                files=files,
                metadatas=request.metadatas,
                document_ids=request.document_ids,
                user=auth_user,
            )
