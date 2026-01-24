import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const CSV_PATH = path.join(process.cwd(), "data", "execution_records.csv");
const CSV_HEADER = "date,recorded_at,tqqq_shares,tqqq_price,sgov_shares,sgov_price,note\n";

async function ensureCsvExists() {
  const dir = path.dirname(CSV_PATH);
  try {
    await fs.access(dir);
  } catch {
    await fs.mkdir(dir, { recursive: true });
  }
  
  try {
    await fs.access(CSV_PATH);
  } catch {
    await fs.writeFile(CSV_PATH, CSV_HEADER, "utf8");
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { executionDate, recordedAt, fills, prices, note } = body;
    
    await ensureCsvExists();
    
    // Build CSV row with prices
    const tqqqShares = fills?.TQQQ ?? 0;
    const tqqqPrice = prices?.TQQQ ?? 0;
    const sgovShares = fills?.SGOV ?? 0;
    const sgovPrice = prices?.SGOV ?? 0;
    const escapedNote = (note || "").replace(/"/g, '""').replace(/\n/g, " ");
    
    const row = `${executionDate},${recordedAt},${tqqqShares},${tqqqPrice},${sgovShares},${sgovPrice},"${escapedNote}"\n`;
    
    await fs.appendFile(CSV_PATH, row, "utf8");
    
    return NextResponse.json({ success: true, path: CSV_PATH });
  } catch (error) {
    console.error("CSV save error:", error);
    return NextResponse.json({ success: false, error: String(error) }, { status: 500 });
  }
}

export async function GET() {
  try {
    await ensureCsvExists();
    const content = await fs.readFile(CSV_PATH, "utf8");
    
    return new NextResponse(content, {
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": "attachment; filename=execution_records.csv",
      },
    });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
