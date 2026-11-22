# Windows, no GPU: mock engine in a private venv (needs only aiohttp numpy soxr).
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) {
    python -m venv .venv
    .\.venv\Scripts\python -m pip install aiohttp "numpy<2" soxr
}
.\.venv\Scripts\python -m translation_server.server --mock @args
