#!/usr/bin/env python3
"""
Gerador de vídeos curtos com IA (até 60 segundos)

Pipeline:
  1. Claude (API Anthropic)  -> cria o roteiro em cenas (narração + prompt de imagem)
  2. Stable Diffusion        -> gera uma imagem por cena (local, via diffusers)
  3. edge-tts                -> narração em voz neural (gratuito, sem chave)
  4. moviepy                 -> junta tudo, com zoom suave e legenda

Instalação:
  pip install anthropic diffusers transformers accelerate torch pillow edge-tts moviepy
  (precisa do ffmpeg instalado no sistema: https://ffmpeg.org)

Uso:
  export ANTHROPIC_API_KEY="sua-chave"
  python gerador_video_ia.py "Curiosidades sobre o espaço" --saida video.mp4
  python gerador_video_ia.py "Dicas de segurança na internet" --formato vertical --voz pt-BR-AntonioNeural

Observação: com GPU NVIDIA (CUDA) cada imagem leva ~1s. Em CPU leva de 30s a 2min cada.
"""

import argparse
import asyncio
import json
import os
import re
import tempfile
import textwrap

import edge_tts
import torch
from anthropic import Anthropic
from diffusers import AutoPipelineForText2Image
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont

DURACAO_MAX = 60.0  # segundos
MODELO_CLAUDE = "claude-sonnet-5-5"
MODELO_IMAGEM = "stabilityai/sdxl-turbo"
FORMATOS = {"vertical": (1080, 1920), "horizontal": (1920, 1080)}


# ---------------------------------------------------------------- 1. Roteiro
def gerar_roteiro(tema: str, idioma: str = "português do Brasil") -> list[dict]:
    """Pede ao Claude um roteiro em cenas. Retorna [{narracao, prompt_imagem}, ...]."""
    client = Anthropic()  # lê ANTHROPIC_API_KEY do ambiente
    instrucao = f"""Crie o roteiro de um vídeo curto sobre: "{tema}".
Regras:
- Entre 6 e 8 cenas. A narração TOTAL deve ter no máximo 130 palavras (cabe em ~55s).
- "narracao": texto falado, em {idioma}, frases curtas e envolventes.
- "prompt_imagem": descrição visual da cena EM INGLÊS, detalhada, sem texto escrito na imagem.
- Responda APENAS com JSON válido neste formato:
{{"cenas": [{{"narracao": "...", "prompt_imagem": "..."}}]}}"""

    resp = client.messages.create(
        model=MODELO_CLAUDE,
        max_tokens=2000,
        messages=[{"role": "user", "content": instrucao}],
    )
    texto = resp.content[0].text
    bruto = re.search(r"\{.*\}", texto, re.S)
    if not bruto:
        raise RuntimeError(f"Resposta sem JSON: {texto[:200]}")
    cenas = json.loads(bruto.group(0))["cenas"]
    if not cenas:
        raise RuntimeError("Roteiro vazio.")
    return cenas


# ---------------------------------------------------------------- 2. Imagens
def gerar_imagens(cenas: list[dict], pasta: str, tamanho: tuple[int, int]) -> list[str]:
    usa_gpu = torch.cuda.is_available()
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODELO_IMAGEM,
        torch_dtype=torch.float16 if usa_gpu else torch.float32,
        variant="fp16" if usa_gpu else None,
    )
    pipe = pipe.to("cuda" if usa_gpu else "cpu")

    caminhos = []
    for i, cena in enumerate(cenas):
        prompt = cena["prompt_imagem"] + ", cinematic lighting, highly detailed, 4k"
        print(f"[imagem {i + 1}/{len(cenas)}] {cena['prompt_imagem'][:70]}...")
        img = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
                   width=512, height=512).images[0]
        img = cobrir(img, tamanho)
        caminho = os.path.join(pasta, f"cena_{i}.png")
        img.save(caminho)
        caminhos.append(caminho)
    del pipe
    return caminhos


def cobrir(img: Image.Image, tamanho: tuple[int, int]) -> Image.Image:
    """Redimensiona e corta a imagem para preencher o quadro (estilo 'cover')."""
    w, h = tamanho
    escala = max(w / img.width, h / img.height)
    img = img.resize((int(img.width * escala) + 1, int(img.height * escala) + 1), Image.LANCZOS)
    esq, topo = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((esq, topo, esq + w, topo + h))


# ---------------------------------------------------------------- 3. Narração
async def _tts(texto: str, voz: str, rate: str, saida: str):
    await edge_tts.Communicate(texto, voz, rate=rate).save(saida)


def gerar_audios(cenas, pasta, voz, rate="+0%") -> list[str]:
    caminhos = []
    for i, cena in enumerate(cenas):
        caminho = os.path.join(pasta, f"voz_{i}.mp3")
        asyncio.run(_tts(cena["narracao"], voz, rate, caminho))
        caminhos.append(caminho)
    return caminhos


def duracao_total(audios: list[str]) -> float:
    total = 0.0
    for a in audios:
        c = AudioFileClip(a)
        total += c.duration
        c.close()
    return total


# ---------------------------------------------------------------- 4. Montagem
def fonte(tamanho: int) -> ImageFont.FreeTypeFont:
    candidatas = [
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidatas:
        if os.path.exists(c):
            return ImageFont.truetype(c, tamanho)
    return ImageFont.load_default()


def legenda_png(texto: str, tamanho: tuple[int, int], pasta: str, idx: int) -> str:
    """Cria um PNG transparente com a legenda (evita depender do ImageMagick)."""
    w, h = tamanho
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f = fonte(int(h * 0.032))
    linhas = textwrap.wrap(texto, width=28 if h > w else 50)
    alt_linha = f.size + 10
    y = int(h * 0.80) - (len(linhas) * alt_linha) // 2
    for linha in linhas:
        largura = draw.textlength(linha, font=f)
        x = (w - largura) / 2
        draw.rectangle([x - 16, y - 4, x + largura + 16, y + f.size + 6], fill=(0, 0, 0, 150))
        draw.text((x, y), linha, font=f, fill="white")
        y += alt_linha
    caminho = os.path.join(pasta, f"legenda_{idx}.png")
    img.save(caminho)
    return caminho


def montar_video(cenas, imagens, audios, tamanho, saida, pasta, fps=24):
    clipes = []
    for i, (cena, img, aud) in enumerate(zip(cenas, imagens, audios)):
        audio = AudioFileClip(aud)
        dur = audio.duration + 0.4  # pequena pausa entre cenas

        # zoom suave (efeito Ken Burns)
        fundo = (ImageClip(img).with_duration(dur)
                 .resized(lambda t, d=dur: 1.0 + 0.08 * (t / d))
                 .with_position("center"))
        legenda = ImageClip(legenda_png(cena["narracao"], tamanho, pasta, i)).with_duration(dur)

        cena_clip = CompositeVideoClip([fundo, legenda], size=tamanho).with_audio(audio)
        clipes.append(cena_clip)

    video = concatenate_videoclips(clipes, method="compose")
    if video.duration > DURACAO_MAX:
        video = video.subclipped(0, DURACAO_MAX)
    print(f"Renderizando {video.duration:.1f}s -> {saida}")
    video.write_videofile(saida, fps=fps, codec="libx264", audio_codec="aac", threads=4)


# ---------------------------------------------------------------- Principal
def main():
    ap = argparse.ArgumentParser(description="Gera vídeos de até 1 minuto com IA.")
    ap.add_argument("tema", help="Assunto do vídeo")
    ap.add_argument("--saida", default="video.mp4")
    ap.add_argument("--formato", choices=FORMATOS, default="vertical")
    ap.add_argument("--voz", default="pt-BR-FranciscaNeural",
                    help="Ex.: pt-BR-AntonioNeural, en-US-AriaNeural")
    args = ap.parse_args()

    tamanho = FORMATOS[args.formato]
    with tempfile.TemporaryDirectory() as pasta:
        print("1/4 Criando roteiro...")
        cenas = gerar_roteiro(args.tema)

        print("2/4 Gerando narração...")
        audios = gerar_audios(cenas, pasta, args.voz)
        total = duracao_total(audios) + 0.4 * len(cenas)
        if total > DURACAO_MAX - 1:  # acelera a fala para caber em 1 minuto
            fator = total / (DURACAO_MAX - 2)
            rate = f"+{int((fator - 1) * 100) + 1}%"
            print(f"Narração com {total:.0f}s; acelerando {rate} para caber em 1 min.")
            audios = gerar_audios(cenas, pasta, args.voz, rate)

        print("3/4 Gerando imagens...")
        imagens = gerar_imagens(cenas, pasta, tamanho)

        print("4/4 Montando vídeo...")
        montar_video(cenas, imagens, audios, tamanho, args.saida, pasta)

    print(f"Pronto: {args.saida}")


if __name__ == "__main__":
    main()
