"""
Site (Gradio) do Gerador de Vídeos com IA.
O usuário digita o tema, clica no botão e recebe o vídeo pronto.

Depende do arquivo gerador_video_ia.py (na mesma pasta).
A chave da Anthropic vem da variável de ambiente ANTHROPIC_API_KEY
(no Hugging Face: Settings -> Variables and secrets -> New secret).
"""

import os
import tempfile

import gradio as gr

import gerador_video_ia as g

VOZES = {
    "Francisca (feminina, BR)": "pt-BR-FranciscaNeural",
    "Antonio (masculina, BR)": "pt-BR-AntonioNeural",
    "Thalita (feminina, BR)": "pt-BR-ThalitaMultilingualNeural",
}


def criar_video(tema, formato, voz, progresso=gr.Progress()):
    if not tema or not tema.strip():
        raise gr.Error("Escreva o tema do vídeo.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise gr.Error("Chave ANTHROPIC_API_KEY não configurada no servidor.")

    tamanho = g.FORMATOS[formato.lower()]
    voz_id = VOZES[voz]
    pasta = tempfile.mkdtemp()
    saida = os.path.join(pasta, "video.mp4")

    progresso(0.05, desc="Criando o roteiro...")
    cenas = g.gerar_roteiro(tema.strip())

    progresso(0.20, desc="Gerando a narração...")
    audios = g.gerar_audios(cenas, pasta, voz_id)
    total = g.duracao_total(audios) + 0.4 * len(cenas)
    if total > g.DURACAO_MAX - 1:
        fator = total / (g.DURACAO_MAX - 2)
        audios = g.gerar_audios(cenas, pasta, voz_id, f"+{int((fator - 1) * 100) + 1}%")

    progresso(0.35, desc="Gerando as imagens (esta é a etapa mais demorada)...")
    imagens = g.gerar_imagens(cenas, pasta, tamanho)

    progresso(0.85, desc="Montando o vídeo...")
    g.montar_video(cenas, imagens, audios, tamanho, saida, pasta)

    return saida


with gr.Blocks(title="Gerador de Vídeos com IA") as site:
    gr.Markdown(
        "# 🎬 Gerador de Vídeos com IA\n"
        "Escreva um tema, clique em **Criar vídeo** e receba um vídeo de até 1 minuto "
        "com roteiro, imagens, narração e legendas feitos por IA."
    )
    tema = gr.Textbox(
        label="Sobre o que é o vídeo?",
        placeholder="Ex.: 5 curiosidades sobre o espaço",
        lines=2,
    )
    with gr.Row():
        formato = gr.Radio(["Vertical", "Horizontal"], value="Vertical",
                           label="Formato (vertical = TikTok/Reels/Shorts)")
        voz = gr.Dropdown(list(VOZES), value="Francisca (feminina, BR)", label="Voz")
    botao = gr.Button("Criar vídeo", variant="primary")
    video = gr.Video(label="Seu vídeo")
    botao.click(criar_video, [tema, formato, voz], video)

if __name__ == "__main__":
    site.queue(max_size=5).launch()
