"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, Command, PieChart, Globe, ClipboardList, BarChart3, Settings, Bell } from "lucide-react";
import { useSettingsStore } from "@/lib/stores/settings-store";

export default function AppHeader() {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const pathname = usePathname();
  const simulationMode = useSettingsStore((s) => s.simulationMode);
  const devScenario = useSettingsStore((s) => s.devScenario);

  const navItems = [
    { name: "Command", href: "/command", icon: Command },
    { name: "Portfolio", href: "/portfolio", icon: PieChart },
    { name: "Macro", href: "/macro", icon: Globe },
    { name: "Records", href: "/records", icon: ClipboardList },
    { name: "Notifications", href: "/notifications", icon: Bell },
    { name: "Analysis", href: "/analysis", icon: BarChart3 },
    { name: "Settings", href: "/settings", icon: Settings },
  ];

  const toggleDrawer = () => setIsDrawerOpen(!isDrawerOpen);
  const closeDrawer = () => setIsDrawerOpen(false);

  return (
    <>
      {/* Top Header Bar */}
      <header className="fixed top-0 left-0 right-0 h-14 bg-inset border-b border-border z-50 flex items-center px-4">
        {/* Hamburger Button */}
        <button
          onClick={toggleDrawer}
          className="w-10 h-10 flex items-center justify-center rounded-lg hover:bg-inset transition-colors"
          aria-label="메뉴 열기"
        >
          <Menu size={22} className="text-muted" />
        </button>

        {/* Logo */}
        <Link
          href="/command"
          className="ml-3 flex items-center gap-2 text-lg font-bold text-white hover:text-accent transition-colors"
        >
          <span className="font-mono tracking-tight">200TQ <span className="text-accent">α</span></span>
        </Link>

        {/* Mode Badges */}
        <div className="ml-3 flex items-center gap-1.5">
          {simulationMode && (
            <span className="text-[11px] font-bold text-choppy bg-choppy-tint px-1.5 py-0.5 rounded border border-choppy/30">
              SIM
            </span>
          )}
          {devScenario && (
            <span className="text-[11px] font-bold text-purple-400 bg-purple-950/50 px-1.5 py-0.5 rounded border border-purple-900/50">
              DEV
            </span>
          )}
        </div>
      </header>


      {/* Backdrop */}
      {isDrawerOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 transition-opacity"
          onClick={closeDrawer}
        />
      )}

      {/* Drawer */}
      <nav
        className={`fixed top-0 left-0 h-full w-64 bg-inset border-r border-border z-50 transform transition-transform duration-300 ease-out ${
          isDrawerOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Drawer Header */}
        <div className="h-14 flex items-center justify-between px-4 border-b border-border">
          <Link
            href="/command"
            onClick={closeDrawer}
            className="flex items-center gap-2 text-lg font-bold text-white"
          >
            <span className="font-mono tracking-tight">200TQ <span className="text-accent">α</span></span>
          </Link>
          <button
            onClick={closeDrawer}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-inset transition-colors"
            aria-label="메뉴 닫기"
          >
            <X size={18} className="text-muted" />
          </button>
        </div>

        {/* Navigation Items */}
        <div className="p-4 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname.startsWith(item.href);
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={closeDrawer}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                  isActive
                    ? "bg-inset text-white font-medium"
                    : "text-muted hover:text-white hover:bg-inset/50"
                }`}
              >
                <Icon size={20} strokeWidth={isActive ? 2.5 : 2} />
                <span className="text-sm">{item.name}</span>
              </Link>
            );
          })}
        </div>

        {/* Footer with Swipe Handle */}
        <div className="absolute bottom-0 left-0 right-0 border-t border-border">
          {/* Mobile Swipe Handle */}
          <div 
            className="py-3 flex justify-center cursor-pointer sm:hidden"
            onClick={closeDrawer}
          >
            <div className="w-12 h-1 rounded-full bg-inset hover:bg-inset/80 transition-colors" />
          </div>
          <div className="px-4 pb-4 pt-2 sm:py-4">
            <div className="text-xs text-muted text-center">
              E03 Strategy Dashboard
            </div>
          </div>
        </div>
      </nav>
    </>
  );
}
