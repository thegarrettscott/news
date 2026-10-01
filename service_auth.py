"""Default-deny inbound service authentication; provider callbacks use separate keys."""
import hmac
import json
import os

class ServiceAuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope.get('type')
        if kind not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        method = scope.get('method', '')
        if kind == 'http' and method in ('GET', 'HEAD') and scope.get('path') in ('/health', '/healthz'):
            return await self.respond(send, 200, {'status': 'ok'}, method == 'HEAD')
        expected = os.environ.get('BACKEND_SECRET', '')
        supplied = [v for k, v in scope.get('headers', []) if k.lower() == b'x-backend-auth']
        valid = bool(expected) and len(supplied) == 1 and hmac.compare_digest(supplied[0], expected.encode())
        if not valid:
            if kind == 'websocket':
                return await send({'type': 'websocket.close', 'code': 4401})
            return await self.respond(send, 401 if expected else 503,
                                      {'error': 'Unauthorized' if expected else 'Service authentication unavailable'}, method == 'HEAD')
        await self.app(scope, receive, send)

    @staticmethod
    async def respond(send, status, data, head=False):
        body = json.dumps(data, separators=(',', ':')).encode()
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store'),
                                (b'content-length', str(len(body)).encode())]})
        await send({'type': 'http.response.body', 'body': b'' if head else body})
