import type { Metadata } from "next";
import "@/styles/globals.css";

export const metadata: Metadata = {
  title: "VALKYRIE. Suspicious Money Flow Investigation Agent.",
  description: "An AI-powered autonomous compliance agent that runs exploration, detects structuring patterns, traces layering chains and Personalised PageRank on 200,000 transactions.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full scroll-smooth">
      <body className="min-h-full flex flex-col bg-[#F2F0EB]">
        {children}
      </body>
    </html>
  );
}
