// Optional password gate for private deployments. When SITE_PASSWORD is set, every request
// must carry HTTP Basic credentials (any user name, that password); when it is not set, the
// site is open. The public project has no password; the private one (with the demo track) does.
import { NextResponse, type NextRequest } from "next/server";

export function proxy(request: NextRequest) {
  const password = process.env.SITE_PASSWORD;
  if (!password) return NextResponse.next();
  const header = request.headers.get("authorization") ?? "";
  if (header.startsWith("Basic ")) {
    try {
      const decoded = atob(header.slice(6));
      const provided = decoded.slice(decoded.indexOf(":") + 1);
      if (provided === password) return NextResponse.next();
    } catch {
      // fall through to the challenge
    }
  }
  return new NextResponse("Private deployment. Enter the password.", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="saturday-night", charset="UTF-8"' },
  });
}

export const config = {
  // Gate everything except Next's own static chunks (they carry no content of their own).
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
