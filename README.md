# Gerador de Vídeos com IA

Gera vídeos de até 1 minuto a partir de um tema:

1. **Claude** cria o roteiro em cenas
2. **Stable Diffusion** gera uma imagem por cena
3. **edge-tts** cria a narração
4. **moviepy** monta o vídeo com zoom suave e legendas

## Requisitos

- Python 3.10+
- [ffmpeg](https://ffmpeg.org) instalado no sistema
- Chave da API da Anthropic
- Recomendado: GPU NVIDIA (em CPU funciona, mas é lento)

## Instalação

```bash
git clone https://github.com/SEU_USUARIO/gerador-video-ia.git
cd gerador-video-ia
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
export ANTHROPIC_API_KEY="sua-chave"   # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sua-chave"
python gerador_video_ia.py "Curiosidades sobre o espaço" --saida video.mp4
```

Opções:

| Opção | Descrição | Padrão |
|---|---|---|
| `--saida` | Nome do arquivo de saída | `video.mp4` |
| `--formato` | `vertical` (1080x1920) ou `horizontal` (1920x1080) | `vertical` |
| `--voz` | Voz do edge-tts (ex.: `pt-BR-AntonioNeural`) | `pt-BR-FranciscaNeural` |

