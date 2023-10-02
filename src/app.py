from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from uuid import uuid4
from arel import HotReload
from arel import Path as ArelPath
import os
from pydantic import BaseModel
from typing import Annotated

from transformers import GPT2Tokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding = "left"

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")





if _debug := os.getenv("DEBUG"):
    hot_reload = HotReload(paths=[ArelPath(".")])
    app.add_websocket_route("/hot-reload", route=hot_reload, name="hot-reload")
    app.add_event_handler("startup", hot_reload.startup)
    app.add_event_handler("shutdown", hot_reload.shutdown)
    templates.env.globals["DEBUG"] = _debug
    templates.env.globals["hot_reload"] = hot_reload


class Parameters(BaseModel):
    model: str = "gpt2"
    content: str = "Once upon a time"
    max_new_tokens: int = 4
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: int = 50
    repetition_penalty: float = 1.2


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "user_id": str(uuid4())}
    )


@app.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    return templates.TemplateResponse(
        "experiment.html", {"request": request, "user_id": str(uuid4())}
    )

@app.get("/compare", response_class=HTMLResponse)
async def compare(request: Request):
    return templates.TemplateResponse(
        "compare.html", {"request": request, "user_id": str(uuid4())}
    )


async def inference(p: Parameters):
    num_return_sequences = 10
    for _ in range(p.max_new_tokens):
        inputs = tokenizer([p.content.strip().replace("\n", "")], return_tensors="pt")
        generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            num_beams=1,
            penalty_alpha=1.1,
            temperature=p.temperature,
            top_k=p.top_k,
            top_p=p.top_p,
            repetition_penalty=p.repetition_penalty,
            num_return_sequences=num_return_sequences,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs = model.generate(**inputs, generation_config=generation_config)
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        word = {}
        for generated_token, transition_score in zip(generated_tokens, transition_scores):
            
            token_id = generated_token[0]
            token = tokenizer.decode(token_id)
            probs = round(np.exp(transition_score[0].numpy()) * 100, 2)
            word[token] = probs

        word = dict(sorted(word.items(), key=lambda x: x[1], reverse=True))

        last_word = (next(iter(word)))
        p.content += last_word

        yield f"""
        <div class="relative group inline-block bg-tranparent rounded hover:border cursor-pointer"><span name="contentText">{next(iter(word))}</span><div class="absolute left-0 mt-2 w-32 bg-background-800 z-50 border border-gray-300 rounded shadow-lg opacity-0 group-hover:opacity-100 transition ease-in-out duration-200">
        """

        for k, v in word.items():
            yield f"""<div class="pb-1">{k} ({v}%)</div>"""
        
        yield "</div></div>"



def test():
    for _ in range(10):
        yield b'Yo '

@app.post("/stream")
async def stream(parameters: Parameters):
    return StreamingResponse(inference(parameters), media_type="text/event-stream")


    # return StreamingResponse(fake_video_streamer(
    #     models, content, maximumLength, temperature, topP, topK, repetitionPenalty
    # ))
    # models: Annotated[str, Form()],
    # content: Annotated[str, Form()],
    # maximumLength: Annotated[int, Form()],
    # temperature: Annotated[float, Form()],
    # topP: Annotated[float, Form()],
    # topK: Annotated[int, Form()],
    # repetitionPenalty: Annotated[float, Form()],