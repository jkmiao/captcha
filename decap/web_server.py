#!/usr/bin/env python
# encoding: utf-8

import tornado.httpserver
import tornado.options
import tornado.web
from tornado.options import define,options
import sys, os, re, random
from apicode import Apicode

reload(sys)
sys.setdefaultencoding('utf-8')


# 定义默认调用端口为8088
define("port",default=8088,help="run on the given port",type=int)

# 验证码演示demo
class VcodeHandler(tornado.web.RequestHandler):
    # get 请求
    def get(self):
        result = self.get_argument("result",{})
        type = self.get_argument('type', u'英文数字')
        self.render("input_code.html", type=type, result=result, source='')

    # post 请求
    def post(self):
        
        result = {'code':0 , 'msg': 'success', 'result': ''}
        
        # 获取参数
        type = self.get_argument('type', 'GNE')
        code_len = self.get_argument('code_len', 0)
        print 'code_len', code_len
        detail = eval(self.get_argument('detail', False))
        imgUrl = self.get_argument('imgUrl', '')
        imgFile = self.request.files.get('imgfile', [])
        fname = ''

        # url 方式
        if imgUrl and re.search('.+\.(jpg|png|bmp|gif)', imgUrl):
            fname = './static/uploads/%s' % imgUrl.split('/')[-1]
            result['result']= vcode.predict_url(type, imgUrl, code_len, detail)
       
        # 上传文件方式
        elif imgFile:
            for img in imgFile:
                with open('./static/uploads/' + img['filename'], 'wb') as fw:
                    fw.write(img['body'])
                fname = './static/uploads/' + img['filename']
                result['result'] = vcode.predict(type, fname, code_len, detail)       
        else:
            errorMsg = "上传验证码图片文件错误或url图片格式不正确"
            result['code'] = '-1'
            result['msg'] = errorMsg
        print 'fname', fname
        self.render("input_code.html", type=type, source=fname, detail=detail, code_len=code_len, result=result)

# 验证码调用api
class VcodeApiHandler(tornado.web.RequestHandler):

    def get(self):
        result = {'code': 1000 , 'msg': u'调用参数错误， 请用post方式请求， type & file 为必填参数', 'result':'xxxx', u'使用说明': u'http://gitlab.tangees.com/miaoweihong/tgcode'}
        json_result = tornado.escape.json_encode(result)
        self.write(json_result)

    def post(self):
        
        result = {'code': 0 , 'msg': 'success', 'result': ''}
        type = self.get_argument('type', 'GNE')  # 验证码类型
        code_len = self.get_argument('code_len', 0)
        detail = self.get_argument('detail', False)

        imgFile = self.request.files.get('imgfile', [])
        if imgFile:
            for img in imgFile:
                fname = './static/uploads/' + 'uploadimg%d.jpg' % (random.randint(1, 50))
                with open(fname, 'wb') as fw:
                    fw.write(img['body'])
                try:
                    result['result'] = vcode.predict(type, fname, code_len, detail)
                except Exception as e:
                    result['code'] = 1001
                    result['msg'] = u'上传文件内容有误' + str(e)
        else:
            result['code'] = 1002
            result['msg'] = u'key=file为空, 没有文件内容'
        # 返回json结果
        json_result = tornado.escape.json_encode(result)
        self.write(json_result)


if __name__ == "__main__":
    
    # 引入自行定义的模块
    vcode = Apicode() 
    tornado.options.parse_command_line()
    
    app = tornado.web.Application(        
        handlers = [(r'/vcode',VcodeHandler), (r'/vcodeapi', VcodeApiHandler), ('/', VcodeHandler)],
        template_path = os.path.join(os.path.dirname(__file__),"templates"),  # 定义视图页面地址，放 html文件
        static_path = os.path.join(os.path.dirname(__file__), "static"),   # 定义静态模板，放 css,js等文件
        debug=True,   # 是否为debug模式
        autoescape=None, # 不设置默认编码
        )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    print "starting tornado at port http://127.0.0.1:%d" % options.port
    tornado.ioloop.IOLoop.instance().start()
