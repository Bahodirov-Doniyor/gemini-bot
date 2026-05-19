import asyncio
import base64
import logging
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import quote

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    ChatPermissions, 
    Message, 
    URLInputFile, 
    BotCommand, 
    ReplyKeyboardRemove, 
    InlineKeyboardMarkup, 
    InlineKeyboardButton,
    CallbackQuery
)
from dotenv import load_dotenv

# ══════════════════════════════════════
# KONFIGURATSIYA
# ══════════════════════════════════════

load_dotenv()

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "20"))
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
OWNER_ID: int = int(os.getenv("OWNER_ID", "0"))
PORT: int = int(os.getenv("PORT", "8080"))

TELEGRAM_MAX_LENGTH = 4096

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    sys.exit("❌ TELEGRAM_TOKEN yoki GEMINI_API_KEY topilmadi! .env faylini tekshiring.")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

# ══════════════════════════════════════
# LOGGING
# ══════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# BOT VA DISPATCHER
# ══════════════════════════════════════

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# ══════════════════════════════════════
# FSM HOLATLARI (STATES)
# ══════════════════════════════════════

class AIForm(StatesGroup):
    waiting_for_imagine = State()
    waiting_for_video = State()
    waiting_for_audio = State()

# ══════════════════════════════════════
# SUHBAT TARIXI (In-Memory)
# ══════════════════════════════════════

conversation_history: dict[int, list[dict]] = {}


def get_history(user_id: int) -> list[dict]:
    return conversation_history.get(user_id, [])


def add_to_history(user_id: int, role: str, parts: list[dict]) -> None:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": role, "parts": parts})
    max_entries = MAX_HISTORY * 2
    if len(conversation_history[user_id]) > max_entries:
        conversation_history[user_id] = conversation_history[user_id][-max_entries:]


def clear_history(user_id: int) -> None:
    conversation_history.pop(user_id, None)


# ══════════════════════════════════════
# YORDAMCHI FUNKSIYALAR
# ══════════════════════════════════════

def split_message(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    return parts


async def send_long_message(message: Message, text: str) -> None:
    for part in split_message(text):
        if part.strip():
            try:
                await message.reply(part, parse_mode="Markdown")
            except Exception:
                await message.reply(part)


async def is_user_admin(chat_id: int, user_id: int) -> bool:
    if user_id == OWNER_ID:
        return True
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ("administrator", "creator")
    except Exception:
        return False


# ══════════════════════════════════════
# GEMINI API
# ══════════════════════════════════════

async def ask_gemini(
    user_id: int,
    user_text: Optional[str],
    media_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> str:
    current_parts: list[dict] = []

    if media_bytes and mime_type:
        encoded = base64.b64encode(media_bytes).decode("utf-8")
        current_parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded,
            }
        })

    text_content = user_text or "Ushbu faylni tahlil qilib, tushuntir."
    current_parts.append({"text": text_content})

    history = get_history(user_id)
    full_contents = history + [{"role": "user", "parts": current_parts}]

    payload = {
        "contents": full_contents,
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.7,
        },
    }

    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            GEMINI_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error("Gemini API xatosi [%d]: %s", resp.status, error_body)
                raise RuntimeError(f"API xatosi: {resp.status}")
            data = await resp.json()

    try:
        ai_text: str = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        logger.error("Kutilmagan API javobi: %s", data)
        raise RuntimeError("API dan kutilmagan javob formati.") from exc

    add_to_history(user_id, "user", [{"text": user_text or "[Media fayl]"}])
    add_to_history(user_id, "model", [{"text": ai_text}])
    return ai_text


# ══════════════════════════════════════
# HEALTH CHECK SERVER
# ══════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format: str, *args) -> None:
        pass


def run_health_server() -> None:
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    logger.info("Health server %d portda ishlamoqda", PORT)
    server.serve_forever()


# ══════════════════════════════════════
# ASOSIY BUYRUQLAR VA MENYU TIZIMI
# ══════════════════════════════════════

@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    name = message.from_user.full_name if message.from_user else "Foydalanuvchi"
    await message.answer(
        f"Salom, <b>{name}</b>! 👋\n\n"
        f"🧠 <b>Men aqlli Gemini AI Botman!</b>\n"
        f"Menga istalgan mavzuda matn yozishingiz, rasm yoki media yuborishingiz mumkin.\n\n"
        f"📖 Buyruqlar haqida batafsil ma'lumot olish uchun: <b>/help</b>\n"
        f"🎛️ Aqlli boshqaruv panelini ochish uchun: <b>/menu</b> yuboring.",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )


@dp.message(Command("help"))
async def cmd_help(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "📖 <b>Bot buyruqlaridan foydalanish qoʻllanmasi:</b>\n\n"
        "🧠 <b>1. AI Buyruqlari:</b>\n"
        "• <code>/imagine [tavsif]</code> — Rasm chizish\n"
        "• <code>/video [tavsif]</code> — Video yaratish\n"
        "• <code>/audio [matn]</code> — Matnni ovoz qilish\n\n"
        "👥 <b>2. Guruh Admin Buyruqlari:</b>\n"
        "• <code>/ban @username</code> — Bloklash\n"
        "• <code>/unban @username</code> — Blokdan olish\n"
        "• <code>/addadmin @username</code> — Admin qilish\n\n"
        "🎛️ Tugmalar orqali toza buyruq berish uchun <b>/menu</b> panelidan foydanlaning.",
        parse_mode="HTML"
    )


@dp.message(Command("menu", "panel"))
async def cmd_menu(message: Message, state: FSMContext) -> None:
    await state.clear()
    
    clean_inline_menu = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🎨 Rasm chizish", callback_data="btn_imagine"),
                InlineKeyboardButton(text="🎬 Video yaratish", callback_data="btn_video")
            ],
            [
                InlineKeyboardButton(text="🔊 Matnni ovoz qilish", callback_data="btn_audio")
            ]
        ]
    )

    await message.answer(
        "🎛️ <b>Tezkor AI boshqaruv paneli:</b>\n\n"
        "Kerakli xizmat tugmasini bosing va hech qanday ortiqcha belgilarsiz toza matn kiritib yuboring! 🔥",
        parse_mode="HTML",
        reply_markup=clean_inline_menu
    )


# ══════════════════════════════════════
# INLINE TUGMALAR HANDLERLARI (CALLBACK)
# ══════════════════════════════════════

@dp.callback_query(F.data == "btn_imagine")
async def process_btn_imagine(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AIForm.waiting_for_imagine)
    await callback.message.answer("🎨 <b>Rasm uchun toza tavsif (prompt) yuboring:</b>\n<i>(Masalan: neon uslubidagi kiber mushuk)</i>", parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "btn_video")
async def process_btn_video(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AIForm.waiting_for_video)
    await callback.message.answer("🎬 <b>Video uchun toza tavsif (prompt) yuboring:</b>", parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "btn_audio")
async def process_btn_audio(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AIForm.waiting_for_audio)
    await callback.message.answer("🔊 <b>Ovozga aylantirish uchun toza matn yuboring:</b>", parse_mode="HTML")
    await callback.answer()


# ══════════════════════════════════════
# FSM MATN QABUL QILISH JORALARI
# ══════════════════════════════════════

@dp.message(AIForm.waiting_for_imagine)
async def ai_imagine_state(message: Message, state: FSMContext) -> None:
    if message.text and message.text.startswith("/"):
        await state.clear()
        return
    await state.clear()
    message.text = f"/imagine {message.text}"
    await cmd_imagine(message)


@dp.message(AIForm.waiting_for_video)
async def ai_video_state(message: Message, state: FSMContext) -> None:
    if message.text and message.text.startswith("/"):
        await state.clear()
        return
    await state.clear()
    message.text = f"/video {message.text}"
    await cmd_video(message)


@dp.message(AIForm.waiting_for_audio)
async def ai_audio_state(message: Message, state: FSMContext) -> None:
    if message.text and message.text.startswith("/"):
        await state.clear()
        return
    await state.clear()
    message.text = f"/audio {message.text}"
    await cmd_audio(message)


# ══════════════════════════════════════
# STANDART BUYRUQLAR
# ══════════════════════════════════════

@dp.message(Command("new"))
async def cmd_new(message: Message, state: FSMContext) -> None:
    await state.clear()
    if message.from_user:
        clear_history(message.from_user.id)
    await message.answer("🔄 Yangi suhbat boshlandi! Kontekst tozalandi.")


@dp.message(Command("model"))
async def cmd_model(message: Message) -> None:
    await message.answer(f"⚙️ Ishchi model: <code>{GEMINI_MODEL}</code>", parse_mode="HTML")


@dp.message(Command("myid"))
async def cmd_myid(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else "Noma'lum"
    await message.answer(f"🆔 Sizning ID: <code>{user_id}</code>", parse_mode="HTML")


@dp.message(Command("info"))
async def cmd_info(message: Message) -> None:
    chat = message.chat
    await message.answer(
        f"📊 <b>Chat ma'lumotlari:</b>\n"
        f"🆔 ID: <code>{chat.id}</code>\n"
        f"📝 Nom: {chat.title or chat.full_name}\n"
        f"📌 Tur: {chat.type}",
        parse_mode="HTML",
    )


# ══════════════════════════════════════
# GURUH MA'MURIYATI MODULLARI
# ══════════════════════════════════════

@dp.message(Command("ban"))
async def cmd_ban(message: Message) -> None:
    if not message.from_user or not await is_user_admin(message.chat.id, message.from_user.id):
        await message.reply("❌ Bu buyruqdan faqat guruh adminlari foydalana oladi.")
        return
    if not message.reply_to_message and len(message.text.split()) < 2:
        await message.answer("❗ Foydalanish: /ban @username yoki xabarga reply qiling")
        return
    try:
        if message.reply_to_message:
            user_id = message.reply_to_message.from_user.id
            name = message.reply_to_message.from_user.full_name
        else:
            username = message.text.split()[1]
            user = await bot.get_chat(username)
            user_id = user.id
            name = user.full_name
        await bot.ban_chat_member(message.chat.id, user_id)
        await message.answer(f"🚫 <b>{name}</b> guruhdan butunlay bloklandi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


@dp.message(Command("unban"))
async def cmd_unban(message: Message) -> None:
    if not message.from_user or not await is_user_admin(message.chat.id, message.from_user.id):
        return
    if len(message.text.split()) < 2:
        await message.answer("❗ Foydalanish: /unban @username")
        return
    try:
        username = message.text.split()[1]
        user = await bot.get_chat(username)
        await bot.unban_chat_member(message.chat.id, user.id)
        await message.answer(f"✅ <b>{user.full_name}</b> blokdan chiqarildi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


@dp.message(Command("addadmin"))
async def cmd_addadmin(message: Message) -> None:
    if not message.from_user or not await is_user_admin(message.chat.id, message.from_user.id):
        return
    if not message.reply_to_message and len(message.text.split()) < 2:
        await message.answer("❗ Foydalanish: /addadmin @username")
        return
    try:
        if message.reply_to_message:
            user_id = message.reply_to_message.from_user.id
            name = message.reply_to_message.from_user.full_name
        else:
            username = message.text.split()[1]
            user = await bot.get_chat(username)
            user_id = user.id
            name = user.full_name
        await bot.promote_chat_member(
            message.chat.id, user_id,
            can_manage_chat=True, can_delete_messages=True, can_restrict_members=True, can_invite_users=True
        )
        await message.answer(f"⭐ <b>{name}</b> yangi admin etib tayinlandi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


@dp.message(Command("post"))
async def cmd_post(message: Message) -> None:
    if not message.from_user or message.from_user.id != OWNER_ID:
        return
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.answer("❗ Foydalanish: /post @kanal Matn")
        return
    try:
        await bot.send_message(parts[1], parts[2])
        await message.answer("✅ Kanalga post yuborildi!")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


# ══════════════════════════════════════
# MULTIMEDIA GENERATSIYA FUNKSIYALARI
# ══════════════════════════════════════

@dp.message(Command("audio"))
async def cmd_audio(message: Message) -> None:
    text_to_speak = (message.text or "").replace("/audio", "").strip()
    if not text_to_speak:
        await message.reply("❗ Matn kiriting. Misol: <code>/audio Salom</code>", parse_mode="HTML")
        return
    status_msg = await message.reply("🔊 Audio tayyorlanmoqda...")
    try:
        encoded_text = quote(text_to_speak)
        audio_url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=uz&client=tw-ob&q={encoded_text}"
        audio_file = URLInputFile(audio_url, filename="audio.mp3")
        await message.reply_audio(audio=audio_file, caption=f"🔊 Ovoz: {text_to_speak[:100]}")
        await status_msg.delete()
    except Exception as e:
        await status_msg.edit_text("❌ Audio generatsiya qilib bo'lmadi.")


@dp.message(Command("imagine"))
async def cmd_imagine(message: Message) -> None:
    prompt = (message.text or "").replace("/imagine", "").strip()
    if not prompt:
        await message.reply("❗ Rasm tavsifini kiriting.", parse_mode="HTML")
        return
    status_msg = await message.reply("🎨 Rasm chizilmoqda...")
    try:
        encoded_prompt = quote(prompt)
        seed = random.randint(1, 99999)
        image_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}"
        photo = URLInputFile(image_url, filename="image.jpg")
        await message.reply_photo(photo=photo, caption=f"🖼 <b>Prompt:</b> {prompt[:200]}", parse_mode="HTML")
        await status_msg.delete()
    except Exception as e:
        await status_msg.edit_text("❌ Rasm yaratishda xatolik yuz berdi.")


@dp.message(Command("video"))
async def cmd_video(message: Message) -> None:
    prompt = (message.text or "").replace("/video", "").strip()
    if not prompt:
        await message.reply("❗ Video tavsifini kiriting.", parse_mode="HTML")
        return
    status_msg = await message.reply("🎬 Video tayyorlanmoqda (15-30 soniya)...")
    try:
        encoded_prompt = quote(prompt)
        video_url = f"https://text-to-video.pollinations.ai/{encoded_prompt}"
        video = URLInputFile(video_url, filename="video.mp4")
        await message.reply_video(video=video, caption=f"🎬 <b>Prompt:</b> {prompt[:200]}", parse_mode="HTML")
        await status_msg.delete()
    except Exception as e:
        await status_msg.edit_text("❌ Video yaratib bo'lmadi.")


# ══════════════════════════════════════
# AI MULTIMODAL HANDLER (KATTA CHATLAR)
# ══════════════════════════════════════

@dp.message(F.content_type.in_({"text", "photo", "voice", "audio", "video", "document"}))
async def multimodal_handler(message: Message, state: FSMContext) -> None:
    if message.text and message.text.startswith("/"):
        return
    if not message.from_user:
        return

    # AGAR FOYDALANUVCHI HOZIR STATE ICHIDA BO'LSA, MULTIMODAL HANDLER ARALASHMAYDI
    current_state = await state.get_state()
    if current_state is not None:
        return

    if message.chat.type in ("group", "supergroup"):
        bot_info = await bot.get_me()
        is_mentioned = message.text and f"@{bot_info.username}" in message.text
        is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == bot_info.id
        if not is_mentioned and not is_reply_to_bot:
            return

    user_id = message.from_user.id
    await bot.send_chat_action(message.chat.id, "typing")

    media_bytes: Optional[bytes] = None
    mime_type: Optional[str] = None
    user_text = message.text or message.caption

    if user_text and message.chat.type in ("group", "supergroup"):
        bot_info = await bot.get_me()
        user_text = user_text.replace(f"@{bot_info.username}", "").strip()

    try:
        if message.photo:
            file_id = message.photo[-1].file_id
            mime_type = "image/jpeg"
            file = await bot.get_file(file_id)
            bio = await bot.download_file(file.file_path)
            media_bytes = bio.read()
        elif message.voice:
            file_id = message.voice.file_id
            mime_type = message.voice.mime_type or "audio/ogg"
            file = await bot.get_file(file_id)
            bio = await bot.download_file(file.file_path)
            media_bytes = bio.read()
        elif message.audio:
            file_id = message.audio.file_id
            mime_type = message.audio.mime_type or "audio/mpeg"
            file = await bot.get_file(file_id)
            bio = await bot.download_file(file.file_path)
            media_bytes = bio.read()
        elif message.video:
            if message.video.file_size and message.video.file_size > 20 * 1024 * 1024:
                await message.reply("❗ Video hajmi 20 MB dan oshmasligi kerak.")
                return
            file_id = message.video.file_id
            mime_type = message.video.mime_type or "video/mp4"
            file = await bot.get_file(file_id)
            bio = await bot.download_file(file.file_path)
            media_bytes = bio.read()

        if not user_text and not media_bytes:
            return

        response = await ask_gemini(user_id, user_text, media_bytes, mime_type)
        await send_long_message(message, response)
    except Exception as e:
        logger.exception("Xato: %s", e)
        await message.reply("❌ Xatolik tufayli javob qaytarib bo'lmadi.")


# ══════════════════════════════════════
# ISHGA TUSHIRISH
# ══════════════════════════════════════

async def main() -> None:
    threading.Thread(target=run_health_server, daemon=True).start()
    
    main_commands = [
        BotCommand(command="start", description="Botni ishga tushirish"),
        BotCommand(command="help", description="📖 Buyruqlarni qo'llash bo'yicha qo'llanma"),
        BotCommand(command="menu", description="🎛️ Toza AI boshqaruv panelini ochish"),
    ]
    try:
        await bot.set_my_commands(main_commands)
        logger.info("Yozish paneli standart menyusi muvaffaqiyatli o'rnatildi.")
    except Exception as cmd_err:
        logger.error(f"Menyuni yuklashda xato: {cmd_err}")

    logger.info("🤖 AI & Admin bot muvaffaqiyatli ishga tushdi!")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
