from __future__ import annotations

from pydantic import BaseModel


class BuildRequest(BaseModel):
    content_path: str
    out_path: str
    title: str
    subtitle: str | None = None
    theme: str = "light"
    preview: bool = False


class BuildResponse(BaseModel):
    success: bool
    pdf_path: str | None = None
    pages: int = 0
    previews: list[str] = []
    assets: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []


class Issue(BaseModel):
    kind: str
    page: int
    severity: str
    detail: str


class ValidateResponse(BaseModel):
    success: bool
    issues: list[Issue] = []


class ChapterSpec(BaseModel):
    id: str
    slug: str
    title: str | None = None
    subtitle: str | None = None
    file: str
    summary: str | None = None


class BookManifest(BaseModel):
    title: str
    subtitle: str | None = None
    theme: str = "light"
    chapters: list[ChapterSpec]


class BookBuildResponse(BuildResponse):
    chapters_rendered: int = 0
    chapters_failed: list[str] = []
    chapter_pages: dict[str, dict] = {}


class InitDetailedResponse(BaseModel):
    success: bool
    run_dir: str
    manifest_path: str
    chapters_dir: str
    chapter_files: list[str]
