import os
import asyncio
from aiohttp import web
import discord
from openai import AsyncOpenAI, APIStatusError, APIConnectionError

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

ALLOWED_CHANNEL_IDS = []  # để trống nếu cho phép mọi kênh
SYSTEM_PROMPT = "Bạn là một trợ lý AI thân thiện, trả lời ngắn gọn, dễ hiểu bằng tiếng Việt."

# Model free trên Mistral La Plateforme (gói "Experiment", rate-limited, không cần thẻ).
# Danh sách đầy đủ: https://docs.mistral.ai/getting-started/models/models_overview/
MODEL_FALLBACK_CHAIN = [
    "mistral-small-latest",
    "open-mistral-7b",
]
MAX_HISTORY = 10

# Cấu hình retry khi API báo lỗi tạm thời (quá tải / rate limit)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRIES_PER_MODEL = 2  # số lần thử lại cho MỖI model trước khi chuyển sang model kế tiếp
BASE_DELAY_SECONDS = 2  # 2s, 4s...

ai_client = AsyncOpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

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
                    print(f"[Mistral] {model_name} lỗi tạm thời ({status_code}), thử lại sau {wait}s "
                          f"(lần {attempt + 1}/{RETRIES_PER_MODEL})...")
                    await asyncio.sleep(wait)
                    continue

                # Hết lượt retry cho model này, hoặc lỗi không thể retry -> chuyển sang model kế tiếp
                print(f"[Mistral] {model_name} thất bại ({status_code}), chuyển sang model kế tiếp...")
                break

            except APIConnectionError as e:
                last_error = e
                if attempt < RETRIES_PER_MODEL - 1:
                    wait = BASE_DELAY_SECONDS * (2 ** attempt)
                    print(f"[Mistral] {model_name} lỗi kết nối, thử lại sau {wait}s "
                          f"(lần {attempt + 1}/{RETRIES_PER_MODEL})...")
                    await asyncio.sleep(wait)
                    continue
                print(f"[Mistral] {model_name} lỗi kết nối liên tục, chuyển sang model kế tiếp...")
                break

            except Exception as e:
                last_error = e
                print(f"[Mistral] {model_name} lỗi không xác định: {e}, chuyển sang model kế tiếp...")
                break

    print(f"[Mistral] Tất cả model đều thất bại. Lỗi cuối: {last_error}")
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

    mentioned = bot.user in message.mentions
    if not mentioned and not message.content.startswith("!ai "):
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
