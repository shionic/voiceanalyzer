#!/usr/bin/env python3
"""
Telegram bot: analyze incoming voice/audio and return the same output as CLI.
"""

import asyncio
import json
import os
import tempfile
from html import escape

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from speaker_pipeline import (
    SimilarityHit,
    VoiceMatchOutput,
    VoiceMatchService,
    format_output_text,
)


service = VoiceMatchService()


def _fmt_opt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _hit_name(hit: SimilarityHit) -> str:
    if hit.author and len(hit.author) <= 48:
        return escape(hit.author)
    return f"record#{hit.record_id}"


def format_output_html(result: VoiceMatchOutput) -> str:
    lines: list[str] = [
        "<b>🎙 Voice analysis complete</b>",
        f"<i>File:</i> <code>{escape(result.filename)}</code>",
        f"<i>Duration:</i> <b>{result.duration:.2f}s</b>",
        "",
        "<b>📈 Core metrics</b>",
        f"• Pitch mean: <b>{result.pitch_mean:.2f} Hz</b>",
        f"• Voicing rate: <b>{result.voicing_rate:.2%}</b>",
        f"• Energy mean: <b>{result.energy_mean:.6f}</b>",
        (
            "• Formants: "
            f"F1={_fmt_opt(result.formants_hz.get('f1'))}, "
            f"F2={_fmt_opt(result.formants_hz.get('f2'))}, "
            f"F3={_fmt_opt(result.formants_hz.get('f3'))}, "
            f"F4={_fmt_opt(result.formants_hz.get('f4'))}"
        ),
        "",
        "<b>🧬 X-vector matching</b>",
    ]

    def one_hit(label_emoji: str, label: str, hit: SimilarityHit | None) -> None:
        if hit is None:
            lines.append(f"{label_emoji} <b>{label}:</b> not found ❌")
            return

        def fmt(v: float | None, digits: int = 2) -> str:
            return _fmt_opt(v, digits)

        def pct(v: float | None, digits: int = 2) -> str:
            if v is None:
                return "n/a"
            return f"{v:.{digits}f}%"

        ref_pitch = hit.reference_pitch or {}
        ref_energy = hit.reference_energy or {}
        ref_formants = hit.reference_formants_hz or {}
        ref_spectral = hit.reference_spectral or {}
        d_pitch = hit.diff_pitch_stats or {}
        d_pitch_pct = hit.diff_pitch_stats_pct or {}
        d_formants = hit.diff_formants_hz or {}
        d_formants_pct = hit.diff_formants_pct or {}
        d_spec = hit.diff_spectral or {}
        d_spec_pct = hit.diff_spectral_pct or {}

        lines.extend([
            f"{label_emoji} <b>{label}:</b> {_hit_name(hit)}",
            f"   Similarity: <b>{hit.similarity:.4f}</b> (distance: {hit.cosine_distance:.4f})",
            f"   Record ID: <code>{hit.record_id}</code>",
            f"   Tags: <code>{escape(', '.join(hit.tags) if hit.tags else 'n/a')}</code>",
            f"   Ref duration/source: {fmt(hit.reference_duration)}s / {escape(hit.reference_author_source or 'n/a')}",
            f"   Ref pitch mean/voicing: {fmt(hit.reference_pitch_mean)} Hz / {fmt(hit.reference_voicing_rate * 100 if hit.reference_voicing_rate is not None else None)}%",

            "   <b>Ref pitch stats</b>",
            f"   mean={fmt(ref_pitch.get('mean'))}, std={fmt(ref_pitch.get('std'))}, min={fmt(ref_pitch.get('min'))}, max={fmt(ref_pitch.get('max'))}",
            f"   p5={fmt(ref_pitch.get('p5'))}, p95={fmt(ref_pitch.get('p95'))}, median={fmt(ref_pitch.get('median'))}, voicing={fmt(ref_pitch.get('voicing_rate') * 100 if ref_pitch.get('voicing_rate') is not None else None)}%",

            "   <b>Δ pitch stats (input-ref)</b>",
            f"   Δmean={fmt(d_pitch.get('mean'))} ({pct(d_pitch_pct.get('mean'))}), Δstd={fmt(d_pitch.get('std'))} ({pct(d_pitch_pct.get('std'))})",
            f"   Δmin={fmt(d_pitch.get('min'))} ({pct(d_pitch_pct.get('min'))}), Δmax={fmt(d_pitch.get('max'))} ({pct(d_pitch_pct.get('max'))})",
            f"   Δp5={fmt(d_pitch.get('p5'))} ({pct(d_pitch_pct.get('p5'))}), Δp95={fmt(d_pitch.get('p95'))} ({pct(d_pitch_pct.get('p95'))}), Δmedian={fmt(d_pitch.get('median'))} ({pct(d_pitch_pct.get('median'))})",

            "   <b>Ref energy</b>",
            f"   mean={fmt(ref_energy.get('mean'), 6)}, std={fmt(ref_energy.get('std'), 6)}, min={fmt(ref_energy.get('min'), 6)}, max={fmt(ref_energy.get('max'), 6)}",
            f"   p5={fmt(ref_energy.get('p5'), 6)}, p95={fmt(ref_energy.get('p95'), 6)}, dynamic_range={fmt(ref_energy.get('dynamic_range'), 6)}",
            f"   Δenergy_mean={fmt(hit.diff_energy_mean, 6)} ({pct(hit.diff_energy_mean_pct)})",

            "   <b>Ref formants (Hz)</b>",
            f"   F1={fmt(ref_formants.get('f1'))}, F2={fmt(ref_formants.get('f2'))}, F3={fmt(ref_formants.get('f3'))}, F4={fmt(ref_formants.get('f4'))}",
            "   <b>Δ formants (Hz, input-ref)</b>",
            f"   ΔF1={fmt(d_formants.get('f1'))} ({pct(d_formants_pct.get('f1'))}), ΔF2={fmt(d_formants.get('f2'))} ({pct(d_formants_pct.get('f2'))})",
            f"   ΔF3={fmt(d_formants.get('f3'))} ({pct(d_formants_pct.get('f3'))}), ΔF4={fmt(d_formants.get('f4'))} ({pct(d_formants_pct.get('f4'))})",

            "   <b>Ref spectral</b>",
            f"   centroid={fmt(ref_spectral.get('centroid'))}, bandwidth={fmt(ref_spectral.get('bandwidth'))}, rolloff={fmt(ref_spectral.get('rolloff'))}",
            f"   flatness={fmt(ref_spectral.get('flatness'), 6)}, zcr={fmt(ref_spectral.get('zero_crossing_rate'), 6)}, rms={fmt(ref_spectral.get('rms_energy'), 6)}",
            "   <b>Δ spectral (input-ref)</b>",
            f"   Δcentroid={fmt(d_spec.get('centroid'))} ({pct(d_spec_pct.get('centroid'))}), Δbandwidth={fmt(d_spec.get('bandwidth'))} ({pct(d_spec_pct.get('bandwidth'))})",
            f"   Δrolloff={fmt(d_spec.get('rolloff'))} ({pct(d_spec_pct.get('rolloff'))}), Δflatness={fmt(d_spec.get('flatness'), 6)} ({pct(d_spec_pct.get('flatness'))})",
            f"   Δzcr={fmt(d_spec.get('zero_crossing_rate'), 6)} ({pct(d_spec_pct.get('zero_crossing_rate'))}), Δrms={fmt(d_spec.get('rms_energy'), 6)} ({pct(d_spec_pct.get('rms_energy'))})",
        ])

    one_hit("👨", "male reference", result.male_best)
    one_hit("👩", "female reference", result.female_best)

    if result.male_best is not None and result.female_best is not None:
        better = "👨 male" if result.male_best.similarity >= result.female_best.similarity else "👩 female"
        lines.extend([
            "",
            "<b>⚖️ Comparison</b>",
            f"• Similarity gap: <b>{abs(result.male_best.similarity - result.female_best.similarity):.4f}</b>",
            f"• Pipeline gap field: <b>{_fmt_opt(result.male_female_similarity_gap, 4)}</b>",
            f"• Better match: <b>{better}</b>",
        ])

    lines.extend([
        "",
        "<b>📊 Full input features</b>",
        f"• Pitch: mean={result.pitch_mean:.2f}, std={result.pitch_std:.2f}, min={result.pitch_min:.2f}, max={result.pitch_max:.2f}, p5={result.pitch_p5:.2f}, p95={result.pitch_p95:.2f}, median={result.pitch_median:.2f}, voicing={result.voicing_rate:.2%}",
        f"• Energy: mean={result.energy_mean:.6f}, std={result.energy_std:.6f}, min={result.energy_min:.6f}, max={result.energy_max:.6f}, p5={result.energy_p5:.6f}, p95={result.energy_p95:.6f}, dynamic_range={result.energy_dynamic_range:.6f}",
        (
            "• Spectral: "
            f"centroid={_fmt_opt(result.spectral.get('centroid'))}, "
            f"bandwidth={_fmt_opt(result.spectral.get('bandwidth'))}, "
            f"rolloff={_fmt_opt(result.spectral.get('rolloff'))}, "
            f"flatness={_fmt_opt(result.spectral.get('flatness'), 6)}, "
            f"zcr={_fmt_opt(result.spectral.get('zero_crossing_rate'), 6)}, "
            f"rms={_fmt_opt(result.spectral.get('rms_energy'), 6)}"
        ),
        (
            "• Formants (Hz): "
            f"F1={_fmt_opt(result.formants_hz.get('f1'))}, "
            f"F2={_fmt_opt(result.formants_hz.get('f2'))}, "
            f"F3={_fmt_opt(result.formants_hz.get('f3'))}, "
            f"F4={_fmt_opt(result.formants_hz.get('f4'))}"
        ),
        f"• MFCC mean ({len(result.mfcc_mean)}): {escape(', '.join(f'{v:.3f}' for v in result.mfcc_mean))}",
    ])

    lines.append("\n✅ <i>Tip:</i> use <code>/analyze</code> as a reply to any voice/audio message.")
    return "\n".join(lines)


def format_output_full_json_html_chunks(result: VoiceMatchOutput, max_payload_len: int = 3200) -> list[str]:
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    chunks: list[str] = []
    start = 0
    while start < len(payload):
        end = min(start + max_payload_len, len(payload))
        chunks.append(payload[start:end])
        start = end

    total = len(chunks)
    messages: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = "<b>📦 Full raw data</b>\n<i>All available fields from analysis + matching:</i>\n"
        part = f"<i>Part {idx}/{total}</i>\n" if total > 1 else ""
        messages.append(f"{header}{part}<pre>{escape(chunk)}</pre>")
    return messages


def format_output_full_text_html_chunks(result: VoiceMatchOutput, max_payload_len: int = 3200) -> list[str]:
    payload = format_output_text(result)
    chunks: list[str] = []
    start = 0
    while start < len(payload):
        end = min(start + max_payload_len, len(payload))
        chunks.append(payload[start:end])
        start = end

    total = len(chunks)
    messages: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = "<b>📄 Full pipeline text output</b>\n<i>Exact output from speaker_pipeline.format_output_text(...):</i>\n"
        part = f"<i>Part {idx}/{total}</i>\n" if total > 1 else ""
        messages.append(f"{header}{part}<pre>{escape(chunk)}</pre>")
    return messages


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 <b>Hi!</b> Send me a voice/audio message and I will analyze it.\n\n"
        "I return:\n"
        "• 🧬 x-vector extraction\n"
        "• 👨 closest speaker with tag <code>male</code>\n"
        "• 👩 closest speaker with tag <code>female</code>\n\n"
        "You can also reply to an existing audio message with <code>/analyze</code>.",
        parse_mode=ParseMode.HTML,
    )


async def _process_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE, telegram_file_id: str, suffix: str) -> None:
    msg = update.message
    if msg is None:
        return

    await msg.reply_text("⏳ <b>Processing audio...</b> This can take some time.", parse_mode=ParseMode.HTML)

    tg_file = await context.bot.get_file(telegram_file_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    try:
        await tg_file.download_to_drive(custom_path=tmp_path)
        result = service.process_file(tmp_path)
        await msg.reply_text(format_output_html(result), parse_mode=ParseMode.HTML)
        #for chunk in format_output_full_text_html_chunks(result):
        #    await msg.reply_text(chunk, parse_mode=ParseMode.HTML)
        #for chunk in format_output_full_json_html_chunks(result):
        #    await msg.reply_text(chunk, parse_mode=ParseMode.HTML)
    except Exception as exc:
        await msg.reply_text(
            f"❌ <b>Error while processing audio:</b>\n<code>{escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
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


async def analyze_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    replied = msg.reply_to_message
    if replied is None:
        await msg.reply_text(
            "ℹ️ Please <b>reply</b> to a voice or audio message with <code>/analyze</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    if replied.voice is not None:
        await _process_audio_file(update, context, replied.voice.file_id, ".ogg")
        return

    if replied.audio is not None:
        suffix = os.path.splitext(replied.audio.file_name or "audio.wav")[-1] or ".wav"
        await _process_audio_file(update, context, replied.audio.file_id, suffix)
        return

    await msg.reply_text(
        "⚠️ Replied message is not voice/audio. Please reply to a valid audio message.",
        parse_mode=ParseMode.HTML,
    )


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_reply))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))

    app.run_polling()


if __name__ == "__main__":
    main()
