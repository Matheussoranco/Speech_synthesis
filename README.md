# 🎤 Advanced Speech Synthesis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Uma solução avançada e completa para síntese de fala (Text-to-Speech) e clonagem de voz, construída com tecnologias de ponta em deep learning.

## ✨ Características Principais

### 🎯 Funcionalidades Core
- **Síntese de Fala de Alta Qualidade**: Modelos TTS state-of-the-art (YourTTS, Tacotron2, FastSpeech2)
- **Clonagem de Voz**: Clone qualquer voz com apenas 3-10 segundos de áudio
- **Suporte Multi-idiomas**: Inglês, Espanhol, Francês, Alemão e mais
- **Interface Web Moderna**: Interface Gradio intuitiva e responsiva
- **API REST**: Endpoints para integração em aplicações

### 🚀 Melhorias Técnicas
- **Arquitetura Transformer Avançada**: Implementação própria com attention multi-head
- **Sistema de Treinamento Robusto**: Early stopping, gradient clipping, mixed precision
- **Processamento de Texto Inteligente**: Normalização automática, fonemização
- **Cache e Otimizações**: Sistema de cache para inferência rápida
- **Monitoramento Completo**: Logs estruturados, métricas de performance

### 🔧 Funcionalidades Técnicas
- **Múltiplas Arquiteturas**: YourTTS, Tacotron2, FastSpeech2, VITS
- **Vocoders Neurais**: HiFi-GAN, MelGAN, WaveRNN
- **Suporte a GPU**: CUDA, MPS (Apple Silicon), CPU
- **Treinamento Distribuído**: Multi-GPU support
- **Reprodutibilidade**: Seeds fixas, deterministic training
- **Avaliação Automatizada**: Métricas PESQ, STOI, SNR, MOS
- **Pré-processamento Inteligente**: Suporte a múltiplos formatos de dataset
- **Export para Produção**: ONNX, TorchScript, Mobile, Quantização

## 🆕 Novas Funcionalidades

### 📊 Sistema de Avaliação
- **Métricas de Qualidade**: PESQ, STOI, SNR, distorção espectral
- **Benchmark de Performance**: Medição de velocidade de inferência (RTF)
- **Comparação de Modelos**: Avaliação side-by-side
- **Relatórios Automatizados**: Geração de relatórios JSON detalhados

### 🔧 Pré-processamento Avançado
- **Detecção Automática de Formato**: LJSpeech, Common Voice, VCTK, genérico
- **Normalização de Áudio**: Reamostragem, normalização de volume
- **Criação de Espectrogramas**: Mel-spectrograms pré-computados
- **Divisão Inteligente**: Train/validation/test splits configuráveis

### 📦 Export para Produção
- **Múltiplos Formatos**: TorchScript, ONNX, Mobile, Quantizado
- **Pacotes de Deploy**: Criação automática de pacotes completos
- **Scripts de Inferência**: Scripts prontos para produção
- **Otimização de Performance**: Quantização dinâmica e estática

## 📁 Estrutura do Projeto

```
Speech_synthesis/
├── 📁 src/                          # Código principal
│   ├── 🐍 model.py                  # Modelos TTS avançados
│   ├── 🐍 train.py                  # Sistema de treinamento
│   ├── 🐍 data.py                   # Datasets e data loaders
│   ├── 🐍 text_processor.py         # Processamento de texto
│   ├── 🐍 gradio_interface.py       # Interface web moderna
│   ├── 🐍 tts_model.py             # Wrapper para Coqui TTS
│   ├── 🐍 speaker_encoder.py        # Encoder para clonagem
│   ├── 🐍 logging_config.py         # Sistema de logs
│   ├── 🐍 utils.py                  # Utilitários
│   ├── 🐍 infer.py                  # Inferência
│   ├── 🐍 clone.py                  # Clonagem de voz
│   ├── 🐍 vocoder.py               # Vocoders neurais
│   ├── 🐍 evaluate.py              # Sistema de avaliação
│   ├── 🐍 preprocess.py            # Pré-processamento
│   └── 🐍 export.py                # Export para produção
├── 📁 models/                       # Modelos pré-treinados
├── 📁 data/                         # Datasets
├── 📁 notebooks/                    # Jupyter notebooks
├── 📁 tests/                        # Testes automatizados
├── 📁 logs/                         # Logs do sistema
├── 📁 cache/                        # Cache de processamento
├── 🔧 config.yaml                   # Configuração principal
├── 🔧 pyproject.toml               # Configuração do projeto
├── 📋 requirements.txt              # Dependências
├── 🚀 main.py                      # Entry point CLI
└── 📖 README.md                    # Esta documentação
```

## 🚀 Instalação Rápida

### Pré-requisitos
- Python 3.8+ 
- PyTorch 2.0+
- CUDA (opcional, para GPU)

### Instalação das Dependências

```bash
# Clone o repositório
git clone <repository-url>
cd Speech_synthesis

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

# Instale espeak (necessário para fonemização)
# Ubuntu/Debian:
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev

# Windows: Baixe e instale do site oficial
# macOS:
brew install espeak
```

### Instalação para Desenvolvimento

```bash
# Instale em modo desenvolvimento
pip install -e .

# Instale dependências de desenvolvimento
pip install -e ".[dev]"

# Configure pre-commit hooks
pre-commit install
```

## 💻 Uso Rápido

### Interface Web (Recomendado)

```bash
# Inicie a interface web
python src/gradio_interface.py

# Ou use o script principal
python main.py web
```

Acesse: `http://localhost:7860`

### CLI - Síntese de Fala

```bash
```bash
# Síntese básica
python main.py infer \
    --text "Olá, este é um teste de síntese de fala" \
    --output audio_saida.wav

# Com modelo personalizado
python main.py infer \
    --text "Texto para sintetizar" \
    --model models/meu_modelo.pth \
    --config config.yaml \
    --output resultado.wav
```

### CLI - Clonagem de Voz

```bash
# Clone uma voz
python main.py clone \
    --text "Texto para falar com a voz clonada" \
    --reference audio_referencia.wav \
    --output voz_clonada.wav

# Com configurações avançadas
python main.py clone \
    --text "Seu texto aqui" \
    --reference voz_referencia.wav \
    --similarity-threshold 0.8 \
    --output resultado_clonado.wav
```

### CLI - Avaliação de Modelo

```bash
# Benchmark de performance
python main.py evaluate \
    --benchmark \
    --checkpoint-path models/checkpoint.pt \
    --repetitions 10

# Avaliação em dataset
python main.py evaluate \
    --dataset data/test \
    --checkpoint-path models/checkpoint.pt \
    --output-dir evaluation_results
```

### CLI - Pré-processamento

```bash
# Processar dataset LJSpeech
python main.py preprocess \
    --input-dir data/LJSpeech-1.1 \
    --output-dir data/processed \
    --format ljspeech

# Detectar formato automaticamente
python main.py preprocess \
    --input-dir data/raw_dataset \
    --output-dir data/processed \
    --format auto
```

### CLI - Export para Produção

```bash
# Export TorchScript
python main.py export \
    --model-path models/checkpoint.pt \
    --output exports/model.pt \
    --format torchscript

# Export ONNX
python main.py export \
    --model-path models/checkpoint.pt \
    --output exports/model.onnx \
    --format onnx

# Criar pacote completo de deploy
python main.py export \
    --model-path models/checkpoint.pt \
    --output deployment_package \
    --format package \
    --include-formats torchscript,onnx,mobile
```

### Treinamento de Modelo

```bash
# Prepare seus dados
mkdir -p data/train data/val

# Organize como: audio.wav + audio.txt para cada amostra

# Inicie o treinamento
python main.py train \
    --config config.yaml \
    --output-dir ./outputs \
    --device auto

# Retome treinamento
python main.py train \
    --config config.yaml \
    --resume outputs/checkpoint_epoch_10.pth
```

## ⚙️ Configuração Avançada

### Arquivo de Configuração Principal

O arquivo `config.yaml` controla todos os aspectos do sistema:

```yaml
# Configuração do sistema
system:
  device: "auto"  # auto, cpu, cuda, mps
  cache_dir: "./cache"
  log_level: "INFO"

# Configuração do modelo
model:
  type: "AdvancedTTS"  # AdvancedTTS, YourTTS, Tacotron2
  params:
    d_model: 512
    num_heads: 8
    num_encoder_layers: 6
    num_decoder_layers: 6

# Configuração de dados
data:
  sample_rate: 22050
  n_mel_channels: 80
  normalize_audio: true
  trim_silence: true

# Treinamento
training:
  batch_size: 16
  learning_rate: 0.0003
  epochs: 100
  early_stopping_patience: 20

# Interface web
web:
  title: "Sistema de Síntese de Fala"
  max_text_length: 1000
  enable_voice_cloning: true
```

### Configurações de Performance

```yaml
performance:
  enable_fp16: true          # Mixed precision training
  enable_caching: true       # Cache para inferência
  cache_size: 1000          # Número de itens no cache
  batch_inference: true      # Inferência em lotes
```

## 🎯 Exemplos de Uso

### Python API

```python
from src.tts_model import TTSWrapper
from src.speaker_encoder import SpeakerEncoder
from src.text_processor import TextProcessor

# Inicializar componentes
tts = TTSWrapper()
speaker_encoder = SpeakerEncoder()
text_processor = TextProcessor(language="pt")

# Síntese básica
text = "Olá, como você está hoje?"
audio = tts.synthesize(text)
tts.save(audio, "output.wav")

# Clonagem de voz
reference_audio = "voz_referencia.wav"
cloned_audio = tts.synthesize(text, speaker_wav=reference_audio)
tts.save(cloned_audio, "voz_clonada.wav")
```

### Processamento em Lote

```python
import os
from pathlib import Path

# Processar múltiplos textos
texts = [
    "Primeiro texto para síntese",
    "Segundo texto para síntese", 
    "Terceiro texto para síntese"
]

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for i, text in enumerate(texts):
    audio = tts.synthesize(text)
    output_path = output_dir / f"audio_{i:03d}.wav"
    tts.save(audio, str(output_path))
    print(f"Salvo: {output_path}")
```

## 🔬 Treinamento Personalizado

### Preparação dos Dados

1. **Estrutura dos Dados**:
```
data/
├── train/
│   ├── audio_001.wav
│   ├── audio_001.txt
│   ├── audio_002.wav
│   ├── audio_002.txt
│   └── ...
└── val/
    ├── audio_val_001.wav
    ├── audio_val_001.txt
    └── ...
```

2. **Formato dos Arquivos**:
   - **Áudio**: WAV, 22050 Hz, mono
   - **Texto**: UTF-8, uma frase por arquivo
   - **Duração**: 1-20 segundos recomendado

### Script de Treinamento Avançado

```python
from omegaconf import OmegaConf
from src.train import TTSTrainer

# Carregue configuração
config = OmegaConf.load("config.yaml")

# Personalize configurações
config.training.batch_size = 32
config.training.learning_rate = 0.001
config.model.params.d_model = 256

# Inicie treinamento
trainer = TTSTrainer(config)
trainer.train()
```

### Monitoramento do Treinamento

```bash
# TensorBoard
tensorboard --logdir logs/

# Weights & Biases (se configurado)
wandb login
# O treinamento irá automaticamente logar métricas
```

## 🧪 Testes e Qualidade

### Executar Testes

```bash
# Todos os testes
pytest

# Testes específicos
pytest tests/test_model.py
pytest tests/test_data.py

# Com cobertura
pytest --cov=src --cov-report=html
```

### Análise de Código

```bash
# Linting
flake8 src/
black src/
mypy src/

# Ou use pre-commit
pre-commit run --all-files
```

## 📊 Métricas e Avaliação

### Métricas de Qualidade de Áudio
- **MOS (Mean Opinion Score)**: Avaliação subjetiva
- **PESQ**: Qualidade perceptual
- **STOI**: Inteligibilidade
- **Mel Cepstral Distortion**: Similaridade espectral

### Métricas de Performance
- **Real-time Factor**: Velocidade de síntese
- **Memory Usage**: Uso de memória
- **GPU Utilization**: Utilização da GPU

### Exemplo de Avaliação

```python
from src.evaluation import AudioEvaluator

evaluator = AudioEvaluator()

# Avaliar qualidade
scores = evaluator.evaluate_batch(
    generated_audios=["output1.wav", "output2.wav"],
    reference_audios=["ref1.wav", "ref2.wav"]
)

print(f"MOS Score: {scores['mos']:.2f}")
print(f"PESQ Score: {scores['pesq']:.2f}")
```

## 🔧 Solução de Problemas

### Problemas Comuns

1. **Erro de ImportError com torch**:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Erro com espeak**:
```bash
# Linux
sudo apt-get install espeak-ng espeak-ng-data

# macOS
brew install espeak-ng
```

3. **Erro de memória GPU**:
```yaml
# No config.yaml
training:
  batch_size: 8  # Reduza o batch size
performance:
  enable_fp16: true  # Use mixed precision
```

4. **Áudio com ruído**:
```yaml
# No config.yaml
data:
  normalize_audio: true
  trim_silence: true
audio:
  noise_reduction: true
```

### Debug Mode

```bash
# Execute com logs detalhados
python main.py infer \
    --text "Teste" \
    --output test.wav \
    --log-level DEBUG
```

## 🤝 Contribuição

### Como Contribuir

1. **Fork** o repositório
2. **Clone** seu fork
3. **Crie** uma branch para sua feature
4. **Implemente** sua mudança
5. **Teste** suas modificações
6. **Commit** com mensagens claras
7. **Push** para sua branch
8. **Abra** um Pull Request

### Padrões de Código

- **PEP 8** para formatação Python
- **Type hints** obrigatórios
- **Docstrings** para todas as funções públicas
- **Testes** para novas funcionalidades

### Exemplo de Contribuição

```python
def nova_funcionalidade(texto: str, parametro: int = 10) -> str:
    """
    Descrição clara da função.
    
    Args:
        texto: Descrição do parâmetro
        parametro: Descrição com valor padrão
        
    Returns:
        Descrição do retorno
        
    Example:
        >>> resultado = nova_funcionalidade("teste", 5)
        >>> print(resultado)
    """
    # Implementação aqui
    return resultado
```

## 📈 Roadmap

### Versão 1.0 (Atual)
- ✅ Interface Gradio moderna
- ✅ Modelos TTS avançados
- ✅ Sistema de clonagem de voz
- ✅ Treinamento robusto
- ✅ Processamento de texto inteligente

### Versão 1.1 (Próxima)
- 🔄 API REST completa
- 🔄 Suporte a mais idiomas
- 🔄 Modelos de emoção
- 🔄 Streaming de áudio
- 🔄 Mobile deployment

### Versão 2.0 (Futuro)
- 🎯 Real-time voice conversion
- 🎯 Multi-modal synthesis
- 🎯 Advanced prosody control
- 🎯 Neural vocoder improvements
- 🎯 Cloud deployment

## 📝 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **Coqui TTS**: Framework base para TTS
- **Resemblyzer**: Speaker encoder
- **Gradio**: Interface web moderna
- **PyTorch**: Framework de deep learning
- **Comunidade Open Source**: Inspiração e suporte

## 📞 Suporte

- **Issues**: [GitHub Issues](../../issues)
- **Documentação**: Este README e código comentado
- **Exemplos**: Pasta `notebooks/`

---

**Desenvolvido com ❤️ para a comunidade de síntese de fala**

*Última atualização: Julho 2025*
