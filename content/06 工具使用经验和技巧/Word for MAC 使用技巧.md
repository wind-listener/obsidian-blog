---
title: "样式格窗调整顺序"
date: 2025-08-07
draft: false
---

# 样式格窗调整顺序
```VBA
Sub 调整样式顺序()

    Dim myStyle As Style
    Dim SetStyleArr As Variant
    Dim SetStyle As Variant
    Dim NewStyleLevel As Long

    ' 样式优先级设置 (样式名称, 优先级)
    SetStyleArr = Array( _
                  Array("标题 1", 1), _
                  Array("标题 2", 2), _
                  Array("标题 3", 3), _
                  Array("正文文本", 4), _
                  Array("题注", 5))

    ' 将原有优先级 1 级修改为 10 级（可调整）
    NewStyleLevel = 9 '（建议不要使用 10，Word 通常支持 1-9）

    ' 遍历所有样式，将原优先级为 1 的调整为 NewStyleLevel
    For Each myStyle In ActiveDocument.Styles
        If myStyle.Priority = 1 Then myStyle.Priority = NewStyleLevel
    Next myStyle

    ' 重新设置指定样式的优先级
    For Each SetStyle In SetStyleArr
        Call SetStyleOrder(SetStyle(0), SetStyle(1))
    Next SetStyle

    ' 释放对象
    Set myStyle = Nothing

    MsgBox "样式设置完成", vbInformation
End Sub

Private Function SetStyleOrder(ByVal StyleName As String, _
                               ByVal StyleOrder As Long) As Boolean
    '
    ' 设置样式顺序
    ' StyleName：样式名
    ' StyleOrder：样式优先级
    '
    Dim myStyle As Style
    Dim isExistStyle As Boolean

    ' 避免错误影响程序运行
    On Error Resume Next
    Set myStyle = ActiveDocument.Styles(StyleName)
    On Error GoTo 0

    ' 检查样式是否存在
    isExistStyle = Not myStyle Is Nothing

    If isExistStyle Then
        With myStyle
            .Priority = StyleOrder   ' 设置优先级
            .Visibility = False      ' 取消隐藏
            .UnhideWhenUsed = False  ' 取消使用前隐藏
            .QuickStyle = True       ' 更新样式
        End With
    End If

    ' 返回是否成功设置样式
    SetStyleOrder = isExistStyle

    ' 释放对象
    Set myStyle = Nothing
End Function
```