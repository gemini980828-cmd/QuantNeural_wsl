import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

export const dynamic = "force-dynamic";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

interface OcrResult {
  success: boolean;
  tqqq_shares?: number;
  sgov_shares?: number;
  raw_response?: string;
  error?: string;
}

function getAnthropicClient(): Anthropic | null {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new Anthropic({ apiKey });
}

const OCR_PROMPT = `이미지는 삼성증권(Samsung Securities) 앱 또는 웹의 포트폴리오/잔고 스크린샷입니다.
이미지에서 TQQQ와 SGOV의 보유 수량을 찾아주세요.

다음 JSON 형식으로만 응답해주세요:
{"tqqq_shares": <숫자>, "sgov_shares": <숫자>}

규칙:
- 해당 종목이 없으면 0으로 표시
- 숫자만 추출 (쉼표, 단위 제거)
- 소수점 있으면 정수로 반올림
- JSON 외 다른 텍스트 없이 응답`;

export async function POST(req: NextRequest): Promise<NextResponse<OcrResult>> {
  try {
    const anthropic = getAnthropicClient();
    if (!anthropic) {
      return NextResponse.json({
        success: false,
        error: "ANTHROPIC_API_KEY not configured",
      }, { status: 500 });
    }

    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json({
        success: false,
        error: "이미지 파일이 필요합니다",
      }, { status: 400 });
    }

    if (!file.type.startsWith("image/")) {
      return NextResponse.json({
        success: false,
        error: "이미지 파일만 업로드 가능합니다",
      }, { status: 400 });
    }

    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json({
        success: false,
        error: "파일 크기는 10MB 이하여야 합니다",
      }, { status: 400 });
    }

    const arrayBuffer = await file.arrayBuffer();
    const base64Image = Buffer.from(arrayBuffer).toString("base64");

    const mediaType = file.type as "image/jpeg" | "image/png" | "image/gif" | "image/webp";

    const message = await anthropic.messages.create({
      model: "claude-3-5-haiku-20241022",
      max_tokens: 256,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: base64Image,
              },
            },
            {
              type: "text",
              text: OCR_PROMPT,
            },
          ],
        },
      ],
    });

    const textContent = message.content.find((block) => block.type === "text");
    const rawResponse = textContent?.type === "text" ? textContent.text : "";

    let tqqq_shares = 0;
    let sgov_shares = 0;

    try {
      const jsonMatch = rawResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        tqqq_shares = Math.round(Number(parsed.tqqq_shares) || 0);
        sgov_shares = Math.round(Number(parsed.sgov_shares) || 0);
      }
    } catch {
      return NextResponse.json({
        success: false,
        error: "OCR 결과 파싱 실패",
        raw_response: rawResponse,
      });
    }

    return NextResponse.json({
      success: true,
      tqqq_shares,
      sgov_shares,
      raw_response: rawResponse,
    });

  } catch (error) {
    console.error("POST /api/portfolio/ocr error:", error);
    return NextResponse.json({
      success: false,
      error: String(error),
    }, { status: 500 });
  }
}
