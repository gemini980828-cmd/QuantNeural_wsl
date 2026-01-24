# QuantNeural Mobile Ops Dashboard

Holo-style investment operations dashboard for E03 strategy.

## 📱 Features

- **Holo Design System**: Dark glassmorphism, Neon Lime/Cyan accents (#090909)
- **Overview Hero**: Live holdings composition with deviations
- **Signals**: Read-only signal history and details
- **News**: RSS-style intelligence feed
- **Reports**: Robustness and backtest artifact viewer

## 🛠️ Stack

- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS + generic GlassCard
- **Icons**: Lucide React
- **Charts**: Recharts
- **Database**: Supabase (Schema provided)

## 🚀 Setup

1. **Install Dependencies**

   ```bash
   npm install
   ```

2. **Run Local Dev**

   ```bash
   npm run dev
   ```

3. **Supabase Setup**
   - Create a new project on Supabase.
   - Run the contents of `supabase-schema.sql` in the SQL Editor.
   - (For production) Connect Vercel to Supabase.

## 📂 Structure

- `app/`: Pages (Overview, Signals, News, Reports)
- `components/ui/`: Core design system (GlassCard)
- `components/overview/`: Hero cards
- `supabase-schema.sql`: Database migration

## 🎨 Design Tokens

- **Background**: `#090909`
- **Primary (Lime)**: `#ABF43F`
- **Secondary (Cyan)**: `#3FF4E5`
- **Card Radius**: `20px`
