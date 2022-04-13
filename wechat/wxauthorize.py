import hashlib
import tornado.web


class WxSignatureHandler(tornado.web.RequestHandler):
    """
    微信服务器签名验证, 消息回复

    check_signature: 校验signature是否正确
    """

    def data_received(self, chunk):
        pass

    def get(self):
        signature = self.get_argument('signature')
        timestamp = self.get_argument('timestamp')
        nonce = self.get_argument('nonce')
        echostr = self.get_argument('echostr')

        result = self.check_signature(signature, timestamp, nonce)
        if result:
            self.write(echostr)

    def check_signature(self, signature, timestamp, nonce):
        """校验token是否正确"""
        token = 'test12345'
        L = [timestamp, nonce, token]
        L.sort()
        s = L[0] + L[1] + L[2]
        sha1 = hashlib.sha1(s.encode('utf-8')).hexdigest()
        return sha1 == signature
