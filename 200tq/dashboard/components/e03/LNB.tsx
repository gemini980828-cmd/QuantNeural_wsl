"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Command, History, FileText, Settings, PieChart } from "lucide-react";

export default function LNB() {
  const pathname = usePathname();

  const navItems = [
    { name: "Command", href: "/command", icon: Command },
    { name: "Portfolio", href: "/portfolio", icon: PieChart },
    { name: "History", href: "/history", icon: History },
    { name: "Reports", href: "/reports", icon: FileText },
    { name: "Settings", href: "/settings", icon: Settings },
  ];

  return (
    <nav className="w-16 min-h-screen bg-neutral-900 border-r border-neutral-800 fixed left-0 top-0 flex flex-col items-center py-6 z-50">
      <div className="flex flex-col gap-6">
        {navItems.map((item) => {
          const isActive = pathname.startsWith(item.href);
          const Icon = item.icon;
          
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`w-10 h-10 flex items-center justify-center rounded-xl transition-colors ${
                isActive
                  ? "bg-neutral-800 text-white"
                  : "text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800/50"
              }`}
              aria-label={item.name}
            >
              <Icon size={20} strokeWidth={isActive ? 2.5 : 2} />
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
