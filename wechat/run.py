import os
import sys

import tornado.httpserver
import tornado.ioloop
import tornado.web
from tornado.options import define, options
from wechat.url import urlpatterns

define('port', default=80, help='run on the given port', type=int)


class Application(tornado.web.Application):
    def __init__(self):
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "wechat/template"),
            static_path=os.path.join(os.path.dirname(__file__), "wechat/static"),
            debug=True,
            login_url='/login',
            cookie_secret='MuG7xxacQdGPR7Svny1OfY6AymHPb0H/t02+I8rIHHE=',
        )
        super(Application, self).__init__(urlpatterns, **settings)


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
