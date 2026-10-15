# Estúdio de Voz

Aplicativo local para gerar fala, clonar e desenhar vozes, transcrever, dublar vídeos e
criar audiolivros. Tudo roda no seu computador (GPU NVIDIA quando disponível).

| Aba | O que faz |
|---|---|
| **Falar** | Texto → fala. A voz pode ser automática, uma voz salva, descrita (gênero, idade, tom, sotaque, sussurro) ou clonada de um áudio de 3–15 s. Textos longos (capítulos, livros em .txt) são divididos e mantêm a mesma voz do início ao fim. Saída em MP3, WAV, OGG ou FLAC. |
| **Vozes** | Salva vozes a partir de um arquivo ou do microfone. A transcrição da referência é feita sozinha. Qualquer áudio gerado também pode virar uma voz salva. |
| **Transcrever** | Áudio/vídeo → texto, com legendas `.srt`/`.vtt`. Também traduz para inglês. |
| **Dublar** | Vídeo/áudio → transcreve, traduz, regrava cada fala com a voz original clonada (ou uma voz salva), ajusta ao tempo de cada fala e devolve o vídeo dublado com legenda. |
| **Histórico** | Tudo o que foi gerado fica salvo em `data/`. |

## Rodar

Requisitos: [uv](https://docs.astral.sh/uv/) e [ffmpeg](https://ffmpeg.org/) no PATH.
Para traduzir dublagens para idiomas além do inglês: [Ollama](https://ollama.com) com qualquer modelo
(ex.: `ollama pull qwen3.5:9b`).

```bash
cd studio
uv sync
uv run python -m server
```

Abre em <http://127.0.0.1:7860>. No Windows, basta dar dois cliques em `start.bat`.

Janela desktop (opcional):

```bash
cd studio/desktop
npm install
npm start
```

Na primeira geração o modelo de voz (~2,5 GB) é baixado do Hugging Face. Na primeira transcrição, o
Whisper `turbo` (~1,5 GB).

## Configuração

Variáveis de ambiente (todas opcionais):

| Variável | Padrão | |
|---|---|---|
| `STUDIO_PORT` | `7860` | Porta do servidor |
| `STUDIO_HOST` | `127.0.0.1` | Use `0.0.0.0` só se quiser acesso pela rede |
| `STUDIO_DATA_DIR` | `studio/data` | Onde ficam vozes, histórico e arquivos |
| `STUDIO_DEVICE` | `auto` | `cuda` ou `cpu` |
| `STUDIO_ASR_MODEL` | `turbo` | Tamanho do Whisper: `tiny`, `base`, `small`, `medium`, `turbo`, `large-v3` |
| `STUDIO_OLLAMA_MODEL` | primeiro instalado | Modelo usado na tradução |
| `STUDIO_REF_MAX_SECONDS` | `15` | Duração máxima da referência de voz |

## API

Documentação interativa em `/api/docs`. Também há endpoints compatíveis com a API de áudio da OpenAI,
para usar com SDKs e outros programas:

```bash
curl http://127.0.0.1:7860/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "Olá, mundo", "voice": "Narrador", "response_format": "mp3"}' -o ola.mp3
```

`voice` é o nome (ou id) de uma voz salva; qualquer outro valor usa a voz automática.
`POST /v1/audio/transcriptions` aceita `file`, `language` e `response_format` (`json`, `text`, `srt`, `vtt`, `verbose_json`).

## Modelos usados

- Síntese: [OmniVoice](https://github.com/k2-fsa/OmniVoice) (Apache-2.0), 600+ idiomas.
- Reconhecimento: [Whisper](https://github.com/openai/whisper) (MIT).
- Tradução: qualquer modelo local via Ollama.

Use clonagem de voz apenas com vozes que você tem permissão para usar.
