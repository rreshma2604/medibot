import "./globals.css";

export const metadata = {
  title: "MediBot — MediAssist Health Network",
  description: "Internal clinical and operational assistant with role-based access",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
