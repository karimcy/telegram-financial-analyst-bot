# Telegram Financial Analyst Bot

This project is a Telegram bot that acts as a financial research analyst, powered by OpenAI's GPT models through the CAMEL AI agent framework. It can answer questions about stocks, provide investment analysis, and fetch real-time stock data using `yfinance`.

## Features

*   **Conversational AI**: Interacts with users in a conversational manner, maintaining context for each user.
*   **Financial Analysis**: Leverages OpenAI's GPT-4o (or other configured models) to provide financial insights.
*   **Real-time Stock Data**: Automatically fetches and incorporates current stock prices, P/E ratios, and market cap for ticker symbols mentioned by the user (powered by `yfinance`).
*   **Secure API Key Management**: Uses a `.env` file to store API keys securely, which is not committed to version control.
*   **Easy to Set Up**: Designed for straightforward setup and execution.

## Prerequisites

*   Python 3.9 or higher
*   Git
*   A Telegram Bot Token
*   An OpenAI API Key

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/karimcy/telegram-financial-analyst-bot.git
    cd telegram-financial-analyst-bot
    ```

2.  **Create and Activate a Virtual Environment**:
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys**:
    *   Create a new file named `.env` in the root directory of the project.
    *   You can copy the example file: `cp .env.example .env` (on macOS/Linux) or `copy .env.example .env` (on Windows).
    *   Open the `.env` file and add your actual API keys:
        ```env
        TELEGRAM_BOT_TOKEN="YOUR_ACTUAL_TELEGRAM_BOT_TOKEN"
        OPENAI_API_KEY="YOUR_ACTUAL_OPENAI_API_KEY"
        ```
    *   **Important**: The `.env` file is included in `.gitignore` and will not be committed to your repository, keeping your keys private.

5.  **Ensure `yfinance` can access market data**:
    Sometimes, `yfinance` might be blocked or require specific configurations depending on your network or region. If you encounter issues with fetching stock data, ensure your environment allows outbound requests as needed by `yfinance`.

## Running the Bot

Once you have completed the setup, you can run the bot using the following command:

```bash
python main.py
```

The bot will start, clear any existing webhook configurations, and begin polling for messages on Telegram. You can interact with it by sending messages to your bot on the Telegram app.

## Project Structure

*   `main.py`: The main script for the Telegram bot.
*   `requirements.txt`: Lists all Python dependencies.
*   `.env`: (You create this) Stores your API keys.
*   `.env.example`: An example template for the `.env` file.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: This file, providing an overview and setup instructions.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change. 