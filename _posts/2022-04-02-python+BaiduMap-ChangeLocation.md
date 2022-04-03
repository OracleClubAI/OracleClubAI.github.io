---
layout: post
title:  Python+百度地图实现地址转换经纬度
categories: TensorFlow
tags:   Python 数据处理
author: 房彬彬
---

* content
{:toc}


# 前期准备

首先安装需要的库，requests(主要用于获取百度地图API地址)

需要导出的相关结果的话就需要引入pandas、csv库，这里根据自己需要选择即可。

## 安装requests包
可以使用：

```python
pip install requests
```
方式安装本次内容主要的包，
其他包也可以通过此方法来安装，这是最简单的方法
##  在百度地图开放平台申请AK

[AK申请链接](http://lbsyun.baidu.com)

> http://lbsyun.baidu.com

![百度地图开发平台主页](https://img-blog.csdnimg.cn/20201118163954777.png#pic_center)

进入百度地图开发平台页面以后先进行账号注册登录然后点击“开发文档”

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118164139411.png#pic_center)

选择LBS云，再点击下图“获取密钥”，进行“傻瓜式”激活流程就可以了！

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118164215333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nvb2xfYnJlZXplX2Jpbg==,size_16,color_FFFFFF,t_70#pic_center)

申请成功后，就可以使用了！！！
#  接下来进入正题
## 首先， AK使用方法

<kbd>直接上图</kbd>

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020111816483013.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nvb2xfYnJlZXplX2Jpbg==,size_16,color_FFFFFF,t_70#pic_center)

申请成功后，进入控制台按照上图操作”创建应用“

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118164950826.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nvb2xfYnJlZXplX2Jpbg==,size_16,color_FFFFFF,t_70#pic_center)

名称自己随便写就行，应用类型就根据自己的实际情况来选择，像anaconda-jupyter是在浏览器打开的,我们就选择浏览器端，启用服务直接全选就行（不用调用每个服务都重新创建应用）

<kbd>白名单：</kbd>
浏览器端可以直接写<kbd>*</kbd>

其他端，下面也都会有相应提示，实在不懂就问”度娘“

以上操作都完成后，就可以直接复制AK使用了！！！

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118171702530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Nvb2xfYnJlZXplX2Jpbg==,size_16,color_FFFFFF,t_70#pic_center)

##  最后，上代码

下列代码中的<kbd>URL</kbd>访问以及API服务信息，可参考:
[百度地图开发平台--Web服务API--地理编码--服务文档](http://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-geocoding)

> http://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-geocoding

```python
import requests

AK = "百度AK"       #  把复制的AK直接粘贴过来就可以了

def change(name):
    url = 'http://api.map.baidu.com/geocoding/v3/?address=%s&output=json&ak=%s'%(name,AK)
    res = requests.get(url)
    if res.status_code == 200:
        val = res.json()
        if val["status"] == 0:
            retval = {'地址':name,'经度':val['result']['location']['lng'],'纬度':val['result']['location']['lat'],'地区标签':val['result']['level'],'是否精确查找':val['result']['precise']}
        else:
            retval = None
        return retval
    else:
        print('无法获取%s经纬度'%name)


if __name__ == '__main__':
	print(change('故宫博物院'))
```
像我用的是anaconda做相关练习，输入“故宫博物院”

最后输出内容如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118173033428.png#pic_center)

如果想批量转化地址，就直接调用这个方法就可以了
