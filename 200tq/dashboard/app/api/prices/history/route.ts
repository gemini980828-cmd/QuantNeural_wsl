/**
 * Price History API
 * 
 * Returns historical price data from prices_daily table.
 * 
 * @route GET /api/prices/history
 * 
 * Query params:
 * - symbol: TQQQ|QQQ|SPLG|SGOV (required)
 * - from: YYYY-MM-DD (optional)
 * - to: YYYY-MM-DD (optional)
 * - limit: number (default 1300, max 4000)
 * - fields: comma-separated (default: date,close)
 */

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Whitelist of allowed symbols
const ALLOWED_SYMBOLS = ['TQQQ', 'QQQ', 'SPLG', 'SGOV'] as const;
type AllowedSymbol = typeof ALLOWED_SYMBOLS[number];

const DEFAULT_LIMIT = 1300;
const MAX_LIMIT = 4000;

interface PriceBar {
  date: string;
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume?: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  
  // Parse and validate symbol
  const symbol = searchParams.get('symbol')?.toUpperCase() as AllowedSymbol | undefined;
  if (!symbol || !ALLOWED_SYMBOLS.includes(symbol)) {
    return NextResponse.json(
      { error: `Invalid symbol. Allowed: ${ALLOWED_SYMBOLS.join(', ')}` },
      { status: 400 }
    );
  }
  
  // Parse date range
  const from = searchParams.get('from');
  const to = searchParams.get('to');
  
  // Parse limit
  let limit = parseInt(searchParams.get('limit') || String(DEFAULT_LIMIT), 10);
  if (isNaN(limit) || limit < 1) limit = DEFAULT_LIMIT;
  if (limit > MAX_LIMIT) limit = MAX_LIMIT;
  
  // Parse fields
  const fieldsParam = searchParams.get('fields') || 'date,close';
  const requestedFields = fieldsParam.split(',').map(f => f.trim().toLowerCase());
  
  // Validate Supabase config
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    return NextResponse.json({ error: 'Supabase not configured' }, { status: 500 });
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  
  try {
    // Build query - always select date and close at minimum
    const selectFields = ['date', 'close'];
    if (requestedFields.includes('open')) selectFields.push('open');
    if (requestedFields.includes('high')) selectFields.push('high');
    if (requestedFields.includes('low')) selectFields.push('low');
    if (requestedFields.includes('volume')) selectFields.push('volume');
    
    let query = supabase
      .from('prices_daily')
      .select(selectFields.join(','))
      .eq('symbol', symbol);
    
    // Apply date filters if provided
    if (from) {
      query = query.gte('date', from);
    }
    if (to) {
      query = query.lte('date', to);
    }
    
    // Supabase has 1000 row default limit, use pagination to bypass
    // Fetch in batches if limit > 1000
    const PAGE_SIZE = 1000;
    let allBars: PriceBar[] = [];
    let offset = 0;
    
    while (allBars.length < limit) {
      const batchSize = Math.min(PAGE_SIZE, limit - allBars.length);
      
      const { data, error } = await supabase
        .from('prices_daily')
        .select(selectFields.join(','))
        .eq('symbol', symbol)
        .order('date', { ascending: false })
        .range(offset, offset + batchSize - 1);
      
      if (error) {
        throw new Error(`Database query failed: ${error.message}`);
      }
      
      if (!data || data.length === 0) break; // No more data
      
      allBars = allBars.concat(data as unknown as PriceBar[]);
      offset += data.length;
      
      if (data.length < batchSize) break; // Last page
    }
    
    // Reverse to get ASC order (oldest first for charts)
    let bars: PriceBar[] = allBars.reverse();
    
    // Get metadata
    const minDate = bars.length > 0 ? bars[0].date : null;
    const maxDate = bars.length > 0 ? bars[bars.length - 1].date : null;
    
    const response = {
      symbol,
      bars,
      meta: {
        minDate,
        maxDate,
        rowCount: bars.length,
      },
    };
    
    // Set cache headers - data updates once daily
    return NextResponse.json(response, {
      headers: {
        'Cache-Control': 's-maxage=3600, stale-while-revalidate=86400',
      },
    });
    
  } catch (error) {
    console.error('Price history API error:', error);
    return NextResponse.json(
      { error: String(error) },
      { status: 500 }
    );
  }
}
