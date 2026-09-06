import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AT-LP120XUSB · Visor 3D WebGPU (vgpu)",
  description:
    "Bandeja giradisco profesional Audio-Technica AT-LP120XUSB renderizada en tiempo real con WebGPU mediante vgpu: PBR, iluminación de estudio, sombras suaves y bloom.",
};

export const viewport: Viewport = {
  themeColor: "#ececec",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="es" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
