import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, Application, ContextTypes
from inference import generate_joke

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello there! I\'m a joke bot. Write for a topic of joke you want!')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')
    await update.message.reply_text(generate_joke(text))


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

def main():
    app = Application.builder().token('-').build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))


    app.run_polling(poll_interval=6)

if __name__ == '__main__':
    main()






