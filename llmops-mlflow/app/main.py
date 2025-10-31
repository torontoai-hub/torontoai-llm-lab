import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from instrumentation import observe_request, logger
from model_client import chat_once

SERVICE_NAME = os.getenv("SERVICE_NAME", "llm-service")
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://tempo:4318/v1/traces")

provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

app = FastAPI(title="LLM Service")

FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    finalize = observe_request(route="/chat", method="POST")
    with tracer.start_as_current_span("chat_inference") as span:
        span.set_attribute("prompt.length", len(prompt))
        try:
            output = chat_once(prompt)
            span.set_attribute("output.length", len(output))
            finalize(200)
            logger.info("chat_success prompt_len=%s output_len=%s", len(prompt), len(output))
            return JSONResponse({"output": output})
        except Exception as e:
            finalize(500)
            logger.exception("chat_failed error=%s", e)
            return JSONResponse({"error": str(e)}, status_code=500)
