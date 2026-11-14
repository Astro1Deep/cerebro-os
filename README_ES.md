# Cerebro OS

[English](README.md) · [Guía de usuario](docs/GUIA.md) · [Notas de versión](docs/RELEASE_NOTES.md)

Cerebro OS es un entorno local para modelos de lenguaje, composición musical y búsqueda en bibliotecas. Este repositorio contiene el código de la beta portátil **1.5.0-beta.3**. Composer analiza acordes, genera progresiones y trabaja con MIDI sin conexión. Jarvis se conecta a un modelo local para conversar por texto y consultar los índices enlazados de BeatOS.

| Componente | Qué hace |
|---|---|
| Composer | Análisis armónico, escalas y cuadrícula Push, progresiones, detección tonal y archivos MIDI |
| Jarvis | Chat de texto, diagnóstico y consulta de índices de BeatOS |
| Motor de IA | MLX-LM en Apple Silicon u Ollama con un modelo local |
| CLI | Estado, diagnóstico sin conexión, perfiles, conocimiento local y evaluación MLX |
| Telegram | Puente opcional para chats privados con política de acceso explícita |

Empieza con Python 3.12 e instala en una carpeta nueva. La guía incluye comandos para macOS, Linux y Windows, los endpoints, las opciones de conexión y el alcance de la validación.

```sh
python3 install.py --target ~/cerebro-beta
```
