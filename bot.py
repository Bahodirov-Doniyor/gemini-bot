import asyncio
import logging
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
TELEGRAM_MAX_LENGTH = 4096

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    sys.exit("TELEGRAM_TOKEN yoki GEMINI_API_KEY topilmadi!")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot, storage=MemoryStorage())

conversation_history = {}


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass

def run_health_server():
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


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


@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    await message.answer(
        f"Salom, <b>{message.from_user.full_name}</b>!\n\n"
        "Men Gemini AI botman. Istalgan savol bering!\n\n"
        "/new — Yangi suhbat\n/help — Yordam"
    )

@dp.message_handler(commands=["help"])
async def cmd_help(message: types.Message):
    await message.answer(
        "Yordam:\n\n"
        "Istalgan savol yuboring\n"
        "/new — tarixni tozalash\n"
        "/model — joriy model"
    )

@dp.message_handler(commands=["new"])
async def cmd_new(message: types.Message):
    clear_history(message.from_user.id)
    await message.answer("Yangi suhbat boshlandi!")

@dp.message_handler(commands=["model"])
async def cmd_model(message: types.Message):
    await message.answer(f"Model: {GEMINI_MODEL}")

@dp.message_handler()
async def ai_handler(message: types.Message):
    user_id = message.from_user.id
    if not message.text.strip():
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        response = await ask_gemini(user_id, message.text.strip())
        for part in split_message(response):
            await message.reply(part)
    except asyncio.TimeoutError:
        await message.reply("Vaqt tugadi. Qaytadan urinib koring.")
    except Exception as e:
        logger.exception(e)
        await message.reply("Xatolik yuz berdi.")


async def main():
    threading.Thread(target=run_health_server, daemon=True).start()
    logger.info("Bot ishga tushdi!")
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
