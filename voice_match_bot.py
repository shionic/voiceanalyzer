#!/usr/bin/env python3
"""
Telegram bot: analyze incoming voice/audio and return the same output as CLI.
"""

import asyncio
import os
import tempfile

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from speaker_pipeline import VoiceMatchService, format_output_text


service = VoiceMatchService()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a voice message or audio file, and I will analyze it and return:\n"
        "- x-vector extraction\n"
        "- closest speaker with tag male\n"
        "- closest speaker with tag female"
    )


async def _process_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE, telegram_file_id: str, suffix: str) -> None:
    msg = update.message
    if msg is None:
        return

    await msg.reply_text("Processing audio... this can take some time.")

    tg_file = await context.bot.get_file(telegram_file_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    try:
        await tg_file.download_to_drive(custom_path=tmp_path)
        result = service.process_file(tmp_path)
        await msg.reply_text(format_output_text(result))
    except Exception as exc:
        await msg.reply_text(f"Error while processing audio: {exc}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None or msg.voice is None:
        return
    await _process_audio_file(update, context, msg.voice.file_id, ".ogg")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None or msg.audio is None:
        return
    suffix = os.path.splitext(msg.audio.file_name or "audio.wav")[-1] or ".wav"
    await _process_audio_file(update, context, msg.audio.file_id, suffix)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))

    app.run_polling()


if __name__ == "__main__":
    main()
