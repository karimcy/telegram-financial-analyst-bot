import os
import logging
import asyncio
import traceback
import time
import httpx
import requests

from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from camel.agents import ChatAgent
from camel.messages.base import BaseMessage
from camel.types.enums import ModelType, RoleType
from camel.models.openai_model import OpenAIModel

# Load environment variables from a local .env file (useful in development)
load_dotenv()

# Retrieve required API keys from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN environment variable is missing.")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is missing.")

# CAMEL relies on the OpenAI SDK internally, so we expose the key via env var
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Basic logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Prepare the system prompt for the financial analyst agent
AGENT_SYSTEM_MESSAGE = BaseMessage(
    role_name="Financial Analyst",
    role_type=RoleType.ASSISTANT,
    meta_dict=None,
    content="""
You are an expert financial research analyst.
Provide thorough, data-driven investment analysis and summaries.
When information is uncertain or unavailable, state the limitation clearly.
Include key metrics (e.g., P/E ratio, revenue growth, risk factors) when relevant.
Conclude with a concise recommendation and risk disclaimer.
"""
)

# Keep one ChatAgent per Telegram user so each conversation maintains context
user_agents: dict[int, ChatAgent] = {}

def get_agent(user_id: int) -> ChatAgent:
    """Return a cached ChatAgent for a user or create a new one."""
    if user_id not in user_agents:
        try:
            # Create OpenAI model
            openai_model = OpenAIModel(
                model_type=ModelType.GPT_4O,
                model_config_dict=dict(temperature=0.3),
            )
            
            # Initialize the agent with necessary parameters
            user_agents[user_id] = ChatAgent(
                system_message=AGENT_SYSTEM_MESSAGE,
                model=openai_model,
            )
            
            logger.info(f"Created new ChatAgent for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error creating ChatAgent: {e}")
            
            # Try a simpler initialization if the previous one failed
            try:
                logger.info("Attempting fallback initialization...")
                
                # Create the model
                openai_model = OpenAIModel(
                    model_type=ModelType.GPT_4O,
                    model_config_dict={"temperature": 0.3},
                )
                
                # Create a simpler ChatAgent
                user_agents[user_id] = ChatAgent(
                    system_message=AGENT_SYSTEM_MESSAGE,
                    model=openai_model,
                )
                
                logger.info("Fallback initialization succeeded")
                
            except Exception as fallback_error:
                logger.error(f"Even fallback initialization failed: {fallback_error}")
                # Raise the original error as it's likely more informative
                raise e
    
    return user_agents[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "Welcome to the Financial Analyst Bot powered by CAMEL!\n"
        "Send me a question about a stock or investment, and I'll analyze it for you."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await update.message.reply_text("Just send a plain-text question about an investment.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main handler for user messages."""
    if not update.message:
        return  # Ignore non-text updates

    user_id = update.effective_user.id
    query_text = update.message.text.strip()

    logger.info("Received message from %s: %s", user_id, query_text)

    # Send "typing" action to indicate the bot is processing
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Get or create the agent for this user
        agent = get_agent(user_id)

        # Augment prompt with basic real-time data if the query looks like a ticker symbol
        augmented_query = query_text
        if len(query_text.split()) == 1 and query_text.isalpha():
            await update.message.reply_text("Looking up stock data for " + query_text.upper() + "...")
            try:
                # Import yfinance conditionally to avoid dependency issues
                import yfinance as yf
                ticker = yf.Ticker(query_text.upper())
                info = ticker.info
                price = info.get("regularMarketPrice")
                pe = info.get("trailingPE")
                market_cap = info.get("marketCap")
                if price:
                    augmented_query += (
                        f"\n\n[Real-time data] Current price: {price}. "
                        f"P/E: {pe}. Market cap: {market_cap}."
                    )
                    logger.info(f"Added stock data for {query_text.upper()}: price={price}, PE={pe}")
            except Exception as e:
                logger.warning(f"Failed to fetch yfinance data: {e}")
                await update.message.reply_text(f"Note: I couldn't retrieve current market data for {query_text.upper()}, but I'll analyze based on my knowledge.")
                # Continue without stock data if there's an error

        # Build the user message for CAMEL
        user_msg = BaseMessage(
            role_name="User",
            role_type=RoleType.USER,
            meta_dict=None,
            content=augmented_query,
        )

        # Step the agent once to get the assistant's response (with basic error recovery)
        try:
            # Refresh the typing status to show we're still working
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # The agent.step() method in CAMEL 0.1.6.7 returns a tuple:
            # (assistant_response: BaseMessage, terminated: bool, info: Dict[str, Any])
            assistant_response, terminated, info = agent.step(user_msg)
            logger.info(f"Agent step completed. Terminated: {terminated}, Info: {info}")

            # assistant_response is a BaseMessage object. Its content attribute holds the reply.
            reply = assistant_response.content

        except ValueError as ve:
            logger.exception(f"ValueError during agent.step unpacking or processing: {ve}. This might indicate an unexpected return structure from agent.step.")
            reply = "Sorry, there was an issue interpreting the AI's response structure."
        except Exception as e:
            logger.exception(f"General agent error: {e}")
            reply = "Sorry, I encountered an error processing your request. Please try again later."

        # Send the reply back to Telegram
        await update.message.reply_text(reply)
        
    except Exception as outer_e:
        # Catch-all for any other errors that might occur
        logger.exception(f"Unexpected error handling message: {outer_e}")
        await update.message.reply_text(
            "I'm sorry, I'm experiencing technical difficulties at the moment. "
            "Please try again in a few moments."
        )


async def clear_webhook_and_updates():
    """Force clear webhook and pending updates to ensure clean startup."""
    try:
        # Create a temporary bot instance just for clearing
        bot = Bot(token=TELEGRAM_TOKEN)
        
        # First, delete any existing webhook
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Successfully cleared webhook configuration")
        
        # Use a direct API call to getUpdates with a very high offset to mark all updates as read
        # This is a more aggressive approach to ensure no pending updates remain
        async with httpx.AsyncClient() as client:
            api_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"offset": -1, "limit": 1, "timeout": 1}
            
            # Try to get the most recent update to find its ID
            response = await client.post(api_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok") and data.get("result"):
                    # If we have updates, get the most recent update ID and set offset to mark all as read
                    last_update_id = data["result"][-1]["update_id"]
                    await client.post(api_url, params={"offset": last_update_id + 1})
                    logger.info(f"Cleared all pending updates up to ID {last_update_id}")
        
        # Sleep briefly to ensure Telegram servers register the cleanup
        await asyncio.sleep(1)
        
    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}")
        # Don't raise the exception, as we want to continue starting the bot


def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors raised during updates."""
    logger.error(f"Update caused error: {context.error}")
    
    # Extract traceback info
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    
    # Log detailed error information
    logger.error(f"Exception details:\n{tb_string}")
    
    # If we're dealing with a Conflict error, provide more specific logging
    if isinstance(context.error, telegram.error.Conflict):
        logger.error("Conflict detected - multiple bot instances may be running")
    
    # If this was caused by a specific update, try to notify the user
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            update.effective_message.reply_text(
                "Sorry, an error occurred while processing your request."
            )
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")


if __name__ == "__main__":
    # Import required libraries
    import traceback
    import requests
    # Ensure os and time are imported if not already (they are used below)
    import os 
    import time

    # Try to ensure clean startup
    try:
        # Clear the webhook synchronously before we start
        logger.info("Clearing webhook before starting...")
        try:
            # Directly use the REST API to clear webhook
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook",
                params={"drop_pending_updates": True}
            )
            if response.status_code == 200:
                logger.info("Successfully cleared webhook via direct API call")
            else:
                logger.warning(f"Failed to clear webhook: {response.text}")
                
            # Sleep briefly to ensure changes take effect
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error during webhook clearing: {e}")
        
        # Set up the application
        logger.info("Starting Financial Analyst Bot...")
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_cmd))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Add error handler for better logging
        application.add_error_handler(error_handler)
        
        # Start the bot
        logger.info("Starting to poll for updates...")
        application.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        # Print stack trace for critical errors
        traceback.print_exc() 