/**
 * Telegram Send API
 * 
 * POST /api/telegram/send
 * { message: string, chatId?: string }
 * 
 * Sends a message to Telegram using bot token from env.
 * Requires: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in Vercel env
 */

import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

interface TelegramSendRequest {
  message: string;
  chatId?: string; // Optional override
  isTest?: boolean;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json() as TelegramSendRequest;
    const { message, chatId: overrideChatId, isTest } = body;
    
    if (!message) {
      return NextResponse.json({ error: "message required" }, { status: 400 });
    }
    
    const botToken = process.env.TELEGRAM_BOT_TOKEN;
    const chatId = overrideChatId || process.env.TELEGRAM_CHAT_ID;
    
    if (!botToken || !chatId) {
      return NextResponse.json({ 
        success: false, 
        error: "Telegram not configured",
        configured: false,
      }, { status: 400 });
    }
    
    // Format message with emoji based on content
    let formattedMessage = message;
    if (isTest) {
      formattedMessage = `🧪 *테스트 메시지*\n\n${message}\n\n_200TQ Dashboard에서 전송됨_`;
    }
    
    // Send to Telegram Bot API
    const telegramUrl = `https://api.telegram.org/bot${botToken}/sendMessage`;
    
    const telegramRes = await fetch(telegramUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text: formattedMessage,
        parse_mode: "Markdown",
        disable_notification: false,
      }),
    });
    
    const telegramData = await telegramRes.json();
    
    if (!telegramData.ok) {
      console.error("Telegram API error:", telegramData);
      return NextResponse.json({
        success: false,
        error: telegramData.description || "Telegram API error",
        configured: true,
      }, { status: 500 });
    }
    
    return NextResponse.json({
      success: true,
      message_id: telegramData.result?.message_id,
      configured: true,
    });
    
  } catch (error) {
    console.error("POST /api/telegram/send error:", error);
    return NextResponse.json({
      success: false,
      error: String(error),
    }, { status: 500 });
  }
}

/**
 * GET /api/telegram/send - Check configuration status
 */
export async function GET() {
  const botToken = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  
  return NextResponse.json({
    configured: !!(botToken && chatId),
    hasBotToken: !!botToken,
    hasChatId: !!chatId,
  });
}
