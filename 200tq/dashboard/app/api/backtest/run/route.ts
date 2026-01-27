/**
 * Backtest Execution API
 * 
 * Executes Python backtest script and returns results.
 * 
 * @route POST /api/backtest/run
 * 
 * Body (JSON):
 * - strategy: E00-E10 (required)
 * - startDate: YYYY-MM-DD (required)
 * - endDate: YYYY-MM-DD (required)
 * - capital: number in KRW (optional, default 100000000)
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

const PROJECT_ROOT = process.env.PROJECT_ROOT || '/home/juwon/QuantNeural_wsl/200tq';
const SCRIPTS_DIR = path.join(PROJECT_ROOT, 'scripts');

const ALLOWED_STRATEGIES = [
  '200TQ', 'E00', 'E01', 'E02', 'E03', 'E04', 
  'E05', 'E06', 'E07', 'E08', 'E09', 'E10'
] as const;
type Strategy = typeof ALLOWED_STRATEGIES[number];

interface BacktestRequest {
  strategy: Strategy;
  startDate: string;
  endDate: string;
  capital?: number;
}

interface BacktestResult {
  status: 'success' | 'error';
  experiment?: string;
  params?: {
    strategy: string;
    startDate: string;
    endDate: string;
    capital: number;
  };
  metrics?: {
    CAGR: number;
    MDD: number;
    Sharpe: number;
    Sortino: number;
    Calmar: number;
    Final: number;
    FinalValue: number;
    TotalTax: number;
    TradesCount: number;
    TradingDays: number;
  };
  equity?: Array<{ date: string; value: number }>;
  elapsed_seconds?: number;
  message?: string;
}

function isValidDate(dateStr: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(dateStr) && !isNaN(Date.parse(dateStr));
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const body = await request.json() as BacktestRequest;
    
    if (!body.strategy || !body.startDate || !body.endDate) {
      return NextResponse.json(
        { status: 'error', message: 'Missing required fields: strategy, startDate, endDate' },
        { status: 400 }
      );
    }
    
    if (!ALLOWED_STRATEGIES.includes(body.strategy)) {
      return NextResponse.json(
        { status: 'error', message: `Invalid strategy. Allowed: ${ALLOWED_STRATEGIES.join(', ')}` },
        { status: 400 }
      );
    }
    
    if (!isValidDate(body.startDate) || !isValidDate(body.endDate)) {
      return NextResponse.json(
        { status: 'error', message: 'Invalid date format. Use YYYY-MM-DD' },
        { status: 400 }
      );
    }
    
    if (new Date(body.startDate) >= new Date(body.endDate)) {
      return NextResponse.json(
        { status: 'error', message: 'startDate must be before endDate' },
        { status: 400 }
      );
    }
    
    const capital = body.capital || 100000000;
    if (capital < 10000 || capital > 100000000000) {
      return NextResponse.json(
        { status: 'error', message: 'Capital must be between 10,000 and 100,000,000,000 KRW' },
        { status: 400 }
      );
    }
    
    const scriptPath = path.join(SCRIPTS_DIR, 'backtest_api.py');
    
    const result = await new Promise<BacktestResult>((resolve, reject) => {
      const args = [
        scriptPath,
        '--strategy', body.strategy,
        '--start', body.startDate,
        '--end', body.endDate,
        '--capital', String(capital),
      ];
      
      const python = spawn('python3', args, {
        cwd: PROJECT_ROOT,
        timeout: 120000,
      });
      
      let stdout = '';
      let stderr = '';
      
      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      python.on('close', (code) => {
        if (code === 0) {
          try {
            const parsed = JSON.parse(stdout.trim());
            resolve(parsed);
          } catch {
            reject(new Error(`Failed to parse output: ${stdout}`));
          }
        } else {
          reject(new Error(`Python exited with code ${code}: ${stderr || stdout}`));
        }
      });
      
      python.on('error', (err) => {
        reject(new Error(`Failed to spawn Python: ${err.message}`));
      });
    });
    
    return NextResponse.json(result);
    
  } catch (error) {
    console.error('Backtest API error:', error);
    return NextResponse.json(
      { 
        status: 'error', 
        message: error instanceof Error ? error.message : String(error) 
      },
      { status: 500 }
    );
  }
}
