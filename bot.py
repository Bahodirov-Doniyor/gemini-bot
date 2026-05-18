import asyncio
import logging
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, ChatPermissions
from aiogram.exceptions import TelegramBadRequest
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
TELEGRAM_MAX_LENGTH = 4096
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    sys.exit("TELEGRAM_TOKEN yoki GEMINI_API_KEY topilmadi!")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
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

def is_owner(user_id):
    return user_id == OWNER_ID


# ══════════════════════════════════════
# ASOSIY BUYRUQLAR
# ══════════════════════════════════════

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        f"Salom, <b>{message.from_user.full_name}</b>! 👋\n\n"
        "🤖 Men <b>Gemini AI</b> botman!\n\n"
        "📋 <b>Buyruqlar:</b>\n"
        "/help — Barcha buyruqlar\n"
        "/new — Yangi suhbat\n"
        "/model — Joriy model",
        parse_mode="HTML"
    )

@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📋 <b>Barcha buyruqlar:</b>\n\n"
        "🤖 <b>AI:</b>\n"
        "/new — Yangi suhbat boshlash\n"
        "/model — Joriy model\n\n"
        "👥 <b>Guruh/Kanal boshqaruvi:</b>\n"
        "/ban @username — Foydalanuvchini bloklash\n"
        "/unban @username — Blokdan chiqarish\n"
        "/mute @username — Sukut qildirish\n"
        "/unmute @username — Sukutdan chiqarish\n"
        "/kick @username — Guruhdan chiqarish\n"
        "/addadmin @username — Admin qilish\n"
        "/removeadmin @username — Adminlikdan olish\n"
        "/pin — Xabarni pin qilish\n"
        "/unpin — Pin xabarni olish\n"
        "/deltmsg — Xabarni o'chirish\n\n"
        "📢 <b>Kanal:</b>\n"
        "/post [kanal] [matn] — Kanalga post\n"
        "/settitle [kanal] [nom] — Kanal nomini o'zgartirish\n"
        "/setdesc [kanal] [tavsif] — Kanal tavsifini o'zgartirish\n"
        "/invite [kanal] — Taklif havolasi\n\n"
        "📊 <b>Ma'lumot:</b>\n"
        "/info — Chat ma'lumotlari\n"
        "/myid — Mening ID m",
        parse_mode="HTML"
    )

@dp.message(Command("new"))
async def cmd_new(message: Message):
    clear_history(message.from_user.id)
    await message.answer("🔄 Yangi suhbat boshlandi!")

@dp.message(Command("model"))
async def cmd_model(message: Message):
    await message.answer(f"⚙️ Model: <code>{GEMINI_MODEL}</code>", parse_mode="HTML")

@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"🆔 Sizning ID ingiz: <code>{message.from_user.id}</code>", parse_mode="HTML")

@dp.message(Command("info"))
async def cmd_info(message: Message):
    chat = message.chat
    await message.answer(
        f"📊 <b>Chat ma'lumotlari:</b>\n"
        f"🆔 ID: <code>{chat.id}</code>\n"
        f"📝 Nom: {chat.title or chat.full_name}\n"
        f"📌 Tur: {chat.type}",
        parse_mode="HTML"
    )


# ══════════════════════════════════════
# FOYDALANUVCHI BOSHQARUVI
# ══════════════════════════════════════

@dp.message(Command("ban"))
async def cmd_ban(message: Message):
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
        await message.answer(f"🚫 <b>{name}</b> bloklandi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("unban"))
async def cmd_unban(message: Message):
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

@dp.message(Command("mute"))
async def cmd_mute(message: Message):
    if not message.reply_to_message:
        await message.answer("❗ Xabarga reply qiling")
        return
    try:
        user_id = message.reply_to_message.from_user.id
        name = message.reply_to_message.from_user.full_name
        await bot.restrict_chat_member(
            message.chat.id, user_id,
            permissions=ChatPermissions(can_send_messages=False)
        )
        await message.answer(f"🔇 <b>{name}</b> sukut qildirildi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("unmute"))
async def cmd_unmute(message: Message):
    if not message.reply_to_message:
        await message.answer("❗ Xabarga reply qiling")
        return
    try:
        user_id = message.reply_to_message.from_user.id
        name = message.reply_to_message.from_user.full_name
        await bot.restrict_chat_member(
            message.chat.id, user_id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True
            )
        )
        await message.answer(f"🔊 <b>{name}</b> sukutdan chiqarildi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("kick"))
async def cmd_kick(message: Message):
    if not message.reply_to_message:
        await message.answer("❗ Xabarga reply qiling")
        return
    try:
        user_id = message.reply_to_message.from_user.id
        name = message.reply_to_message.from_user.full_name
        await bot.ban_chat_member(message.chat.id, user_id)
        await bot.unban_chat_member(message.chat.id, user_id)
        await message.answer(f"👢 <b>{name}</b> guruhdan chiqarildi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("addadmin"))
async def cmd_addadmin(message: Message):
    if not message.reply_to_message and len(message.text.split()) < 2:
        await message.answer("❗ Foydalanish: /addadmin @username yoki xabarga reply qiling")
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
            can_manage_chat=True,
            can_delete_messages=True,
            can_manage_video_chats=True,
            can_restrict_members=True,
            can_promote_members=False,
            can_change_info=True,
            can_invite_users=True,
            can_pin_messages=True
        )
        await message.answer(f"⭐ <b>{name}</b> admin qilindi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("removeadmin"))
async def cmd_removeadmin(message: Message):
    if not message.reply_to_message:
        await message.answer("❗ Xabarga reply qiling")
        return
    try:
        user_id = message.reply_to_message.from_user.id
        name = message.reply_to_message.from_user.full_name
        await bot.promote_chat_member(message.chat.id, user_id)
        await message.answer(f"❌ <b>{name}</b> adminlikdan olindi!", parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


# ══════════════════════════════════════
# XABAR BOSHQARUVI
# ══════════════════════════════════════

@dp.message(Command("pin"))
async def cmd_pin(message: Message):
    if not message.reply_to_message:
        await message.answer("❗ Xabarga reply qiling")
        return
    try:
        await bot.pin_chat_message(message.chat.id, message.reply_to_message.message_id)
        await message.answer("📌 Xabar pin qilindi!")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("unpin"))
async def cmd_unpin(message: Message):
    try:
        await bot.unpin_chat_message(message.chat.id)
        await message.answer("📌 Pin xabar olindi!")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("deltmsg"))
async def cmd_deltmsg(message: Message):
    if not message.reply_to_message:
        await message.answer("❗ O'chirmoqchi bo'lgan xabarga reply qiling")
        return
    try:
        await bot.delete_message(message.chat.id, message.reply_to_message.message_id)
        await message.delete()
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


# ══════════════════════════════════════
# KANAL BOSHQARUVI
# ══════════════════════════════════════

@dp.message(Command("post"))
async def cmd_post(message: Message):
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.answer("❗ Foydalanish: /post @kanal Matn")
        return
    try:
        channel = parts[1]
        text = parts[2]
        await bot.send_message(channel, text)
        await message.answer("✅ Post yuborildi!")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("settitle"))
async def cmd_settitle(message: Message):
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.answer("❗ Foydalanish: /settitle @kanal Yangi nom")
        return
    try:
        await bot.set_chat_title(parts[1], parts[2])
        await message.answer("✅ Kanal nomi o'zgartirildi!")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("setdesc"))
async def cmd_setdesc(message: Message):
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.answer("❗ Foydalanish: /setdesc @kanal Tavsif")
        return
    try:
        await bot.set_chat_description(parts[1], parts[2])
        await message.answer("✅ Kanal tavsifi o'zgartirildi!")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")

@dp.message(Command("invite"))
async def cmd_invite(message: Message):
    parts = message.text.split()
    if len(parts) < 2:
        await message.answer("❗ Foydalanish: /invite @kanal")
        return
    try:
        link = await bot.create_chat_invite_link(parts[1])
        await message.answer(f"🔗 Taklif havolasi:\n{link.invite_link}")
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


# ══════════════════════════════════════
# AI HANDLER
# ══════════════════════════════════════

@dp.message(F.text)
async def ai_handler(message: Message):
    user_id = message.from_user.id
    if not message.text.strip():
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        response = await ask_gemini(user_id, message.text.strip())
        for part in split_message(response):
            await message.reply(part)
    except asyncio.TimeoutError:
        await message.reply("⏳ Vaqt tugadi.")
    except Exception as e:
        logger.exception(e)
        await message.reply("❌ Xatolik yuz berdi.")


async def main():
    threading.Thread(target=run_health_server, daemon=True).start()
    logger.info("Bot ishga tushdi!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
