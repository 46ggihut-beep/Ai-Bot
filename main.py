import os
import asyncio
from aiohttp import web
import discord
from openai import AsyncOpenAI, APIStatusError, APIConnectionError

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

ALLOWED_CHANNEL_IDS = []  # để trống nếu cho phép mọi kênh
# Các kênh trong danh sách này: bot trả lời MỌI tin nhắn, không cần @bot hay !ai.
# Lấy ID kênh: bật Developer Mode trong Discord (User Settings > Advanced) rồi
# chuột phải/nhấn giữ vào kênh > Copy Channel ID.
FREE_CHAT_CHANNEL_IDS = [123456789012345678]  # thay bằng ID kênh thật, để trống [] nếu không dùng
SYSTEM_PROMPT = (
    "Bạn là một người bạn thân đang nhắn tin, không phải trợ lý AI trang trọng. "
    "Nói chuyện tự nhiên, xưng hô kiểu bạn bè (tao/mày, t/m, hoặc mình/bạn tùy ngữ cảnh người nhắn dùng), "
    "câu trả lời ngắn gọn như nhắn tin thật, không dài dòng, không liệt kê gạch đầu dòng trừ khi thật sự cần thiết. "
    "Hiểu các từ viết tắt, teencode, tiếng lóng tiếng Việt thường dùng khi nhắn tin/chat "
    "(vd: k=không, đc=được, ko=không, vs=với, mn=mọi người, ny=người yêu, sml, vcl, cc, ez, gg, afk, brb...). "
    "Tuyệt đối KHÔNG dùng emoji hay icon trong câu trả lời, chỉ dùng chữ thuần túy. "
    "Không cần lịch sự khách sáo kiểu 'dạ vâng ạ', cứ nói chuyện bình thường như hai người bạn nhắn tin qua lại."
)
current_system_prompt = SYSTEM_PROMPT  # có thể đổi bằng lệnh !setprompt trong Discord, không cần sửa code

# Model free trên Google AI Studio (Gemini), qua endpoint tương thích OpenAI.
# Quota free: 15 request/phút, ~1500 request/ngày cho Flash (có thể đổi tùy thời điểm,
# kiểm tra lại tại: https://ai.google.dev/gemini-api/docs/rate-limits)
MODEL_FALLBACK_CHAIN = [
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
]
MAX_HISTORY = 10

# Cấu hình retry khi API báo lỗi tạm thời (quá tải / rate limit)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRIES_PER_MODEL = 2  # số lần thử lại cho MỖI model trước khi chuyển sang model kế tiếp
BASE_DELAY_SECONDS = 2  # 2s, 4s...

ai_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

chat_history = {}


# --- Web server nhỏ chỉ để Render (free tier) thấy app đang "sống" ---
# UptimeRobot sẽ ping vào đây mỗi vài phút để Render không cho ngủ
async def handle_ping(request):
    return web.Response(text="Bot is running")


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))  # Render tự cấp PORT
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()


async def generate_reply(history):
    """Gọi OpenRouter bất đồng bộ, tự retry rồi chuyển sang model free khác nếu bị quá tải."""
    last_error = None

    messages = [{"role": "system", "content": current_system_prompt}] + history

    for model_name in MODEL_FALLBACK_CHAIN:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                response = await ai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                # Model trả về rỗng -> coi như lỗi, thử tiếp
                last_error = f"Model {model_name} trả về nội dung rỗng"
                break

            except APIStatusError as e:
                last_error = e
                status_code = e.status_code

                if status_code in RETRYABLE_STATUS_CODES and attempt < RETRIES_PER_MODEL - 1:
                    wait = BASE_DELAY_SECONDS * (2 ** attempt)
                    print(f"[Gemini] {model_name} lỗi tạm thời ({status_code}), thử lại sau {wait}s "
                          f"(lần {attempt + 1}/{RETRIES_PER_MODEL})...")
                    await asyncio.sleep(wait)
                    continue

                # Hết lượt retry cho model này, hoặc lỗi không thể retry -> chuyển sang model kế tiếp
                print(f"[Gemini] {model_name} thất bại ({status_code}), chuyển sang model kế tiếp...")
                break

            except APIConnectionError as e:
                last_error = e
                if attempt < RETRIES_PER_MODEL - 1:
                    wait = BASE_DELAY_SECONDS * (2 ** attempt)
                    print(f"[Gemini] {model_name} lỗi kết nối, thử lại sau {wait}s "
                          f"(lần {attempt + 1}/{RETRIES_PER_MODEL})...")
                    await asyncio.sleep(wait)
                    continue
                print(f"[Gemini] {model_name} lỗi kết nối liên tục, chuyển sang model kế tiếp...")
                break

            except Exception as e:
                last_error = e
                print(f"[Gemini] {model_name} lỗi không xác định: {e}, chuyển sang model kế tiếp...")
                break

    print(f"[Gemini] Tất cả model đều thất bại. Lỗi cuối: {last_error}")
    return "Xin lỗi, hiện tại AI đang quá tải hoặc gặp sự cố, bạn thử lại sau ít phút nhé 🙏"


@bot.event
async def on_ready():
    print(f"Bot đã online: {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    # Lệnh đổi tính cách bot ngay trong Discord, không cần sửa code/deploy lại.
    # Cách dùng: !setprompt <mô tả tính cách mới>
    global current_system_prompt
    if message.content.startswith("!setprompt "):
        current_system_prompt = message.content[len("!setprompt "):].strip()
        chat_history.clear()  # xóa lịch sử cũ để tránh lẫn giọng nói cũ/mới
        await message.reply("Đã đổi tính cách bot xong, lịch sử chat cũng được reset.")
        return

    if message.content.strip() == "!resetprompt":
        current_system_prompt = SYSTEM_PROMPT
        chat_history.clear()
        await message.reply("Đã đưa bot về tính cách mặc định.")
        return

    is_free_chat_channel = message.channel.id in FREE_CHAT_CHANNEL_IDS
    mentioned = bot.user in message.mentions

    if not is_free_chat_channel and not mentioned and not message.content.startswith("!ai "):
        return

    user_text = message.content.replace(f"<@{bot.user.id}>", "").replace("!ai ", "").strip()
    if not user_text:
        return

    channel_id = message.channel.id
    history = chat_history.setdefault(channel_id, [])
    history.append({"role": "user", "content": user_text})
    history[:] = history[-MAX_HISTORY:]

    async with message.channel.typing():
        reply_text = await generate_reply(history)

    history.append({"role": "assistant", "content": reply_text})
    history[:] = history[-MAX_HISTORY:]

    await message.reply(reply_text[:2000])


async def main():
    await start_web_server()
    await bot.start(DISCORD_TOKEN)


asyncio.run(main())
