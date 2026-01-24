import AppHeader from "../../components/AppHeader";

export default function ShellLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-e03-bg text-fg">
      <AppHeader />
      <main className="pt-14 w-full">
        <div className="mx-auto w-full max-w-[1440px] px-4 sm:px-6 py-6 lg:py-8">
          {children}
        </div>
      </main>
    </div>
  );
}
