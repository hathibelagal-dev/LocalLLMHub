from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from importlib import resources
from huggingface_hub import hf_hub_download, list_repo_files, get_hf_file_metadata, hf_hub_url
from starlette.responses import StreamingResponse
from fastapi import HTTPException
import logging
import asyncio
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
with resources.path("localllmhub", "templates") as template_dir:
    templates = Jinja2Templates(directory=str(template_dir))

loaded_models = {}
loaded_tokenizers = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    hf_token = os.getenv('HF_TOKEN', '')
    return templates.TemplateResponse("index.html", {
        "request": request,
        "hf_token": hf_token
    })

@app.post("/api/set-token")
async def set_token(request: Request):
    data = await request.json()
    if not data or 'hf_token' not in data:
        raise HTTPException(status_code=400, detail="No token provided")
    
    hf_token = data['hf_token']
    os.environ['HF_TOKEN'] = hf_token
    return {"message": "Token updated successfully"}

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/llm-manager", response_class=HTMLResponse)
async def llm_manager(request: Request):
    return templates.TemplateResponse("llm-manager.html", {"request": request})

@app.get("/downloader", response_class=HTMLResponse)
async def downloader(request: Request):
    return templates.TemplateResponse("downloader.html", {"request": request})

async def download_progress(repo_id: str):
    try:
        files = list_repo_files(repo_id=repo_id)
        essential_files = [f for f in files if f.endswith(('.safetensors', '.json')) and not f.startswith(('pytorch_', 'flax_'))]
        total_size = 0
        downloaded_size = 0
        
        for file in essential_files:
            metadata = get_hf_file_metadata(hf_hub_url(repo_id, file))
            total_size += metadata.size

        yield f"data: {{\"status\": \"initializing\", \"message\": \"Found {len(essential_files)} essential files, total size: {(total_size / (1024 * 1024)):.2f} MB\"}}\n\n"

        for file in essential_files:
            yield f"data: {{\"status\": \"downloading\", \"file\": \"{file}\", \"progress\": {(downloaded_size / total_size * 100):.1f}\"}}\n\n"
            
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                token=os.getenv('HF_TOKEN', None)
            )
            
            file_size = os.path.getsize(file_path)
            downloaded_size += file_size
            
            percentage = (downloaded_size / total_size) * 100
            yield f"data: {{\"status\": \"downloading\", \"file\": \"{file}\", \"progress\": {percentage:.1f}}}\n\n"

        yield "data: {\"status\": \"completed\", \"progress\": 100.0}\n\n"
        
    except Exception as e:
        logger.error(f"Error downloading model {repo_id}: {str(e)}")
        yield f"data: {{\"status\": \"error\", \"message\": \"{str(e)}\"}}\n\n"

@app.get("/api/install-llm")
async def install_llm(name: str):
    if not name or "/" not in name:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    logger.info(f"Starting installation of model: {name}")
    return StreamingResponse(
        download_progress(name),
        media_type="text/event-stream"
    )

@app.get("/api/list-llms")
async def list_llms():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    if not os.path.exists(cache_dir):
        return {"error": f"Cache directory {cache_dir} not found"}
    
    models = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    
    llm_info = []
    
    for model in models:
        model_path = os.path.join(cache_dir, model)        
        total_size = 0
        for dirpath, dirname, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        size_mb = total_size / (1024 * 1024)
        model_name = model.replace("models--", "").replace("--", "/")
        if model_name.startswith("."):
            continue
        llm_info.append({
            "name": model_name,
            "size_mb": round(size_mb, 2)
        })
    
    return {"llms": llm_info}

async def stream_chat(model_name: str, message: str):
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            if loaded_tokenizers[model_name].pad_token is None:
                loaded_tokenizers[model_name].pad_token = loaded_tokenizers[model_name].eos_token
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            yield f"data: {{\"error\": \"Failed to load model: {str(e)}\"}}\n\n"
            return

    model = loaded_models[model_name]
    tokenizer = loaded_tokenizers[model_name]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide concise, direct answers"},
        {"role": "user", "content": message}
    ]

    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{role}: {content}\n"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "streamer": streamer
    }

    import threading
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        escaped_text = json.dumps(new_text)[1:-1]
        yield f"data: {{\"token\": \"{escaped_text}\"}}\n\n"
        await asyncio.sleep(0.05)

    yield "data: {\"status\": \"completed\"}\n\n"

@app.post("/api/chat")
async def chat_with_llm(request: Request):
    data = await request.json()
    model_name = data.get("model")
    message = data.get("message")

    if not model_name or not message:
        raise HTTPException(status_code=400, detail="Model name and message are required")

    return StreamingResponse(
        stream_chat(model_name, message),
        media_type="text/event-stream"
    )