import type { Metadata } from "next";
import "./globals.css";
import { SessionRecovery } from "@/components/SessionRecovery";

export const metadata: Metadata = { title: "Room3D", description: "Turn a room video into an explorable 3D scene." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}<SessionRecovery /></body></html>; }
