import asyncio
import os
import unittest
from unittest.mock import patch
from service_auth import ServiceAuthMiddleware

class AuthTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, path='/news', method='GET', headers=(), kind='http', secret='synthetic-build-secret'):
        events=[];calls=[]
        async def downstream(scope, receive, send):
            calls.append(scope)
            if kind=='http':await send({'type':'http.response.start','status':422,'headers':[]})
        async def receive():raise AssertionError('Denied request body must not be consumed')
        async def send(event):events.append(event)
        with patch.dict(os.environ, {'BACKEND_SECRET':secret}):
            await ServiceAuthMiddleware(downstream)({'type':kind,'method':method,'path':path,'headers':headers},receive,send)
        return events,calls
    async def test_denials_before_handlers(self):
        for method in ['GET','POST','PUT','PATCH','DELETE','OPTIONS','HEAD']:
            for headers in [(),[(b'x-backend-auth',b'wrong')],[(b'x-backend-auth',b'synthetic-build-secret'),(b'x-backend-auth',b'wrong')]]:
                with self.subTest(method=method,headers=headers):
                    events,calls=await self.request(method=method,headers=headers)
                    self.assertEqual(events[0]['status'],401);self.assertFalse(calls)
                    self.assertIn((b'cache-control',b'no-store'),events[0]['headers'])
    async def test_missing_configuration_fails_closed(self):
        events,calls=await self.request(secret='');self.assertEqual(events[0]['status'],503);self.assertFalse(calls)
    async def test_valid_key_reaches_existing_validation(self):
        for method in ['GET','POST']:
            events,calls=await self.request(method=method,headers=[(b'x-backend-auth',b'synthetic-build-secret')])
            self.assertEqual(events[0]['status'],422);self.assertEqual(len(calls),1)
    async def test_only_exact_health_reads_are_public(self):
        for path in ['/health','/healthz']:
            for method in ['GET','HEAD']:
                events,calls=await self.request(path=path,method=method,secret='')
                self.assertEqual(events[0]['status'],200);self.assertFalse(calls)
                if method=='HEAD':self.assertEqual(events[1]['body'],b'')
        for path in ['/health/extra','/docs','/openapi.json','/new-private-route']:
            events,calls=await self.request(path=path);self.assertEqual(events[0]['status'],401);self.assertFalse(calls)
        events,calls=await self.request(path='/health',method='POST');self.assertEqual(events[0]['status'],401)
    async def test_websockets_fail_closed(self):
        events,calls=await self.request(kind='websocket');self.assertEqual(events,[{'type':'websocket.close','code':4401}]);self.assertFalse(calls)
    async def test_lifespan_passthrough(self):
        events,calls=await self.request(kind='lifespan');self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
