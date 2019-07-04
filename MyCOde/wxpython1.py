'''
Created on 30-Jun-2019

@author: Arijeet Mukherjee
'''
import wx
def func1(event):
    print("The buton is pressed....")
app=wx.App()
win=wx.Frame(None,title="myApp",size=(480,480))
panel=wx.Panel(win)
button1=wx.Button(panel,label="click",pos=(200,10),size=(70,30))
button1.Bind(wx.EVT_BUTTON,func1)
win.Show()
app.MainLoop()