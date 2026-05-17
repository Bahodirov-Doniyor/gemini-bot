"""
Gemini AI Telegram Bot — Professional Edition
"""

import asyncio
import logging
import os
import sys

import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
TELEGRAM_MAX_LENGTH = 4096

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    sys.exit("❌ TELEGRAM_TOKEN yoki GEMINI_API_KEY topilmadi!")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

conversation_history: dict[int, list[dict]] = {}


def get_history(user_id):
    return conversation_history.get(user_id, [])


def add_to_history(user_id, role, text):
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": role, "parts": [{"text": text}]})
    if len(conversation_history[user_id]) > MAX_HISTORY * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY * 2:]


def clear_history(user_id):
    conversation_history.pop(user_id, None)


def split_message(text, limit=TELEGRAM_MAX_LENGTH):
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return parts


async def ask_gemini(user_id, user_text):
    add_to_history(user_id, "user", user_text)
    payload = {
        "contents": get_history(user_id),
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.9,
        },
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(GEMINI_URL, json=payload, headers={"Content-Type": "application/json"}) as resp:
            if resp.status != 200:
                raise RuntimeError(f"API xatosi: {resp.status}")
            data = await resp.json()
    try:
        ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("API dan kutilmagan javob.")
    add_to_history(user_id, "model", ai_text)
    return ai_text


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        f"👋 Salom, <b>{message.from_user.full_name}</b>!\n\n"
        "Men <b>Gemini AI</b> botman. Istalgan savol bering! 🧠\n\n"
        "/new — Yangi suhbat\n/help — Yordam",
        parse_mode="HTML",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "🤖 <b>Yordam</b>\n\n"
        "• Istalgan savol yuboring\n"
        "• /new — tarixni tozalash\n"
        "• /model — joriy model",
        parse_mode="HTML",
    )


@dp.message(Command("new"))
async def cmd_new(message: Message):
    clear_history(message.from_user.id)
    await message.answer("🔄 Yangi suhbat boshlandi!")


@dp.message(Command("model"))
async def cmd_model(message: Message):
    await message.answer(f"⚙️ Model: <code>{GEMINI_MODEL}</code>", parse_mode="HTML")


@dp.message(F.text)
async def ai_handler(message: Message):
    user_id = message.from_user.id
    if not message.text.strip():
        return
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        response = await ask_gemini(user_id, message.text.strip())
        for part in split_message(response):
            await message.reply(part)
    except asyncio.TimeoutError:
        await message.reply("⏳ Vaqt tugadi. Qaytadan urinib ko'ring.")
    except Exception as e:
        logger.exception(e)
        await message.reply("❌ Xatolik yuz berdi.")


async def main():
    logger.info("🚀 Bot ishga tushdi!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
