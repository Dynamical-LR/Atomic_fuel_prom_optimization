Public minTimeColumn, maxTimeColumn
Sub очистЧасть()
    'Application.ScreenUpdating = False
    Dim rcell As Range
    Dim rrng As Range
    Лист1.Range("B2:L2000").Interior.Pattern = xlNone


    'поиск сегодняшнего дня и лимитный столбец
    minTimeColumn = 7
    i = 1
    Do While i < 3000
        If Date = Лист2.Cells(1, i) Then
            minTimeColumn = i + 92 ' от текущего времени + 23ч
        End If
        i = i + 1
    Loop
    maxTimeColumn = minTimeColumn + 96 - 1

    Лист2.Range(Лист2.Cells(3, minTimeColumn), Лист2.Cells(48, maxTimeColumn + 100)).ClearContents
    For i = 3 To 39 Step 6
        Лист2.Cells(i + 2, minTimeColumn - 1).Interior.Color = 0
        For ii = minTimeColumn To maxTimeColumn + 100
            If Лист2.Cells(i + 2, ii).Interior.ThemeColor <> xlThemeColorDark2 Then
                Лист2.Cells(i + 2, ii).Interior.Pattern = xlNone
            End If
        Next ii
    Next i
    For i = 46 To 48 Step 2
        For ii = minTimeColumn To maxTimeColumn + 100
            If Лист2.Cells(i, ii).Interior.ThemeColor <> xlThemeColorDark2 Then
                Лист2.Cells(i, ii).Interior.Pattern = xlNone
            End If
        Next ii
    Next i
    Лист2.Range(Лист2.Cells(3, 6), Лист2.Cells(44, 6)).ClearContents 'очистка текущих температур
End Sub

Sub очистПолнБланк()
    Range("F3:XFD100").Interior.Pattern = xlNone 'очистка бланка
    Range("F3:XFD100").ClearContents
End Sub

Sub run()
    'Лист1.Range("A2:Q3000").Interior.Pattern = xlNone 'очистка бланка
    Dim mass As New Dictionary
    Dim iteraziya As Variant
    maxSeri = 0

    очистЧасть

    'серии по группам продолжительности 1, 2, ... часа
    i = 2
    Do While Лист1.Cells(i, 2) <> ""
        ivlGR = Fix(get_ivl(i, 8, 12))
        If ivlGR > maxSeri Then maxSeri = ivlGR
        mass.Add i, ivlGR
        i = i + 2
    Loop

    'Итерации 1, 2, 3 - проход от самой большей по продолжительности серии к меньшей
    iteraziya = Array("prim", "tek", "low", "top")
    For Each statusIter In iteraziya
        For ivlGR = maxSeri To 0 Step -1
            i = 2
            Do While Лист1.Cells(i, 2) <> ""
                If ivlGR = mass.Item(i) Then
                    If Лист1.Cells(i, 1).Interior.Color = 65535 Or Лист1.Cells(i, 1).Interior.Color = 255 Then 'в план помеченные желтым строки
                        res = set_line(i, statusIter) 'отрисовка линии
                    End If
                End If
                If Лист1.Cells(i, 1).Interior.Color = 5296274 Then
                    Лист1.Range(Лист1.Cells(i, 1), Лист1.Cells(i + 1, 12)).Interior.Color = 5296274 'зелёным вся строка если фактич. выполнено
                End If
                i = i + 2
            Loop
        Next ivlGR
    Next
    SaveFile 'сохраняю последний результат
End Sub

Function get_mesto(Npeh, longLine, startT)
    get_mesto = 0
    If longLine > 0 Then
        longLine_ = longLine * 4 '+ 1
        countZero = 0
        For i = startT To maxTimeColumn '+ longLine_
            If Лист2.Cells(Npeh + 2, i).Interior.Pattern = xlNone Then countZero = countZero + 1 Else countZero = 0
            If countZero >= longLine_ Then
                get_mesto = i + 1 - longLine_ 'начало вставки
                Exit For
            End If
        Next i
    End If
End Function

Function set_mesto(Npeh, longLine, OMD, startT, колСадка, txt)
    set_mesto = 0
    If longLine > 0 Then
        If OMD(1) = "ковка" Then
            OMD_ = 46
            longLine_ = longLine * 4
            clrOMD = 5296274 'зеленый
            t1 = "K"
        ElseIf OMD(1) = "прокат" Then
            OMD_ = 48
            longLine_ = longLine * 4
            clrOMD = 10498160 'фиолет
            t1 = "П"
        Else
            If OMD(1) = "отжиг" Then OMD(0) = "отжиг"
            OMD_ = 1
            longLine_ = longLine * 4
        End If
        If OMD(0) = "нагрев" Then
            clrNagr = 5232127 'оранж
        ElseIf OMD(0) = "подогрев" Then
            clrNagr = 65535 'желнтый
        ElseIf OMD(0) = "отжиг" Then
            clrNagr = 1079284 'яркий оранж
        ElseIf OMD(0) = "перевод t" Then
            clrNagr = 12632306 'красный
        End If

        'смещение наложение прокат/ковка с учётом колСадок
        startT_ = startT
        D = 0
        If OMD_ > 1 Then
            Do While Лист2.Cells(OMD_, startT + longLine_).Interior.Pattern <> xlNone _
            Or Лист2.Cells(OMD_, startT + longLine_ + (колСадка - 1)).Interior.Pattern <> xlNone
                startT = startT + 1
                D = D + 1
                If OMD(0) = "подогрев" And D > 1 Then 'перерыв между нагревом и подогревом
                    set_mesto = 0
                    Exit Function
                End If
                If startT + longLine_ + (колСадка - 1) > maxTimeColumn Then
                    set_mesto = 0
                    Exit Function
                End If
            Loop
            Лист2.Range(Лист2.Cells(Npeh + 2, startT_), Лист2.Cells(Npeh + 2, startT)).Interior.Color = 12632256
        End If


        'подписи
        Лист2.Cells(Npeh, startT) = txt(0)
        Лист2.Cells(Npeh + 1, startT) = txt(1)
        Лист2.Cells(Npeh + 2, startT) = txt(2)

        Лист2.Range(Лист2.Cells(Npeh + 2, startT), Лист2.Cells(Npeh + 2, startT + longLine_ - 1)).Interior.Color = clrNagr
        If OMD_ > 1 Then
            For i = 0 To колСадка - 1
                Лист2.Cells(OMD_, startT + longLine_) = 1
                Лист2.Cells(OMD_, startT + longLine_).Interior.Color = clrOMD
                Лист2.Cells(Npeh + 2, startT + longLine_) = t1
                Лист2.Cells(Npeh + 2, startT + longLine_).Interior.Color = clrOMD
                startT = startT + 1
            Next i
        End If
        set_mesto = startT + longLine_
    End If
End Function

Function set_line(rowPul, statusIter)
    Dim txt(5)
    Npeh = 3
    Do While Лист2.Cells(Npeh, 2) <> ""
        'получить первую печь с такой же текущей температурой (строка печи)
        yesPeh = False

        temper = Лист1.Cells(rowPul, 7)
        lowT = 0
        ar = Split(Лист2.Cells(Npeh, 2), ",")
        For i0 = 0 To UBound(ar)
            t = Trim(ar(i0)) * 1
            If t = temper Then
                If Лист2.Cells(Npeh, 6) = "" Then temper_end = Лист2.Cells(Npeh, 5) Else temper_end = Лист2.Cells(Npeh, 6)
                If temper < temper_end Then 'если есть время на перевод на понижение/повышение
                    lowT = 5
                ElseIf temper > temper_end Then
                    lowT = 8
                End If
            End If
        Next i0

        If statusIter = "prim" Then 'вжопегорит
            If Лист1.Cells(rowPul, 1).Interior.Color = 255 Then
                yesPeh = True
            End If
        ElseIf statusIter = "tek" Then
            If temper_end = temper Then yesPeh = True
        ElseIf statusIter = "low" Then 'температуру на понижение
            If t = temper And t < Лист2.Cells(Npeh, 5) Then
                yesPeh = True
            End If
        ElseIf statusIter = "top" Then 'температуру на повышение
            yesPeh = True
        End If

        If rowPul = 74 And Npeh = 27 Then 'And statusIter = "top"
            f = f
        End If

        'моделирую вставку линии и выясняю столбец старта
        If yesPeh = True Then
            longLine = 0 'в часах
            metkaPul = 0
            колСадка = Лист1.Cells(rowPul, 5) 'колСадка
            If Лист2.Cells(Npeh, 6) = "" Then temper_end = Лист2.Cells(Npeh, 5) Else temper_end = Лист2.Cells(Npeh, 6)
            If temper <> temper_end Then 'если есть время на перевод на понижение/повышение
                longLine = longLine + lowT
            End If
            For s = 8 To 12 'по нагревам в серии
                'длинна нагрева
                If Лист1.Cells(rowPul + 1, s) <> "" Then
                    longLine = longLine + get_ivl(rowPul, s, s)
                    longLine = longLine + колСадка / 4
                End If
                If Лист1.Cells(rowPul, s) <> "" Then
                    'сравниваю диаметры поковок
                    Dzag = Split(Лист1.Cells(rowPul, s), "-")(0) * 1
                    Dpeh = Лист2.Cells(Npeh, 4)
                    If Dzag < Dpeh And Dpeh <> "" Then
                        metkaPul = 1
                    End If
                End If
                'проверка - помеченные ранее не беру
                If Лист1.Cells(rowPul, s).Interior.Color = 65535 Then
                    metkaPul = 1
                Else
                    Лист1.Range(Лист1.Cells(rowPul, 2), Лист1.Cells(rowPul + 1, 12)).Interior.Color = 12632306 'пометка в пуле красным
                End If
                'проверка - печи только на ковку на прокат
                aa = get_kovProk(rowPul, s)(1)
                If InStr(1, Лист2.Cells(Npeh, 3), aa) = 0 And aa <> "" Then metkaPul = 1
            Next s
            startT = get_mesto(Npeh, longLine, minTimeColumn)

            'произвожу вставку линии всей серии
            If startT > 0 And metkaPul = 0 Then 'если вся серия влезает
                If temper <> temper_end Then
                    txt(0) = ""
                    txt(1) = "Перевод " & temper_end & "'C->" & temper & "'C" 'формирую надпись перевода t
                    txt(2) = ""
                    Лист2.Cells(Npeh, 6) = temper 'новая текущая температура
                    startT = set_mesto(Npeh, lowT, Array("перевод t", ""), startT, колСадка, txt)
                End If

                'если есть время на нагрев
                For s = 8 To 12
                    txt(0) = Лист1.Cells(rowPul, 2) & ", " & Лист1.Cells(rowPul, 4) 'rowPul & "." &
                    txt(1) = "[" & Лист1.Cells(rowPul, s) & "]" & "Б." & Лист1.Cells(rowPul, 6)
                    txt(2) = temper & "'C " & колСадка & "шт."
                    If startT > 0 Then
                        startT = set_mesto(Npeh, get_ivl(rowPul, s, s), get_kovProk(rowPul, s), startT, колСадка, txt)
                    End If
                    If startT > 0 Then
                        Лист1.Range(Лист1.Cells(rowPul, 2), Лист1.Cells(rowPul + 1, 12)).Interior.Color = 65535 'пометка в пуле жёлтым отрисованных
                    End If
                Next s
                Exit Function
            End If
        End If
        Npeh = Npeh + 6
    Loop
End Function

Function get_ivl(row, s1, s2)
    Sum = 0
    For i = s1 To s2
        v = Лист1.Cells(row + 1, i)
        If v <> "" Then
            If InStr(1, v, "+") > 0 Then
                a = Split(v, "+")
                Sum = Sum * 1 + Replace(a(0), "ч", "") * 1
            Else
                a = Split(v, " ")
                Sum = Sum * 1 + Replace(a(0), "ч", "") * 1
            End If
        End If
    Next i
    get_ivl = Sum
End Function

Function get_kovProk(row, s1)
    get_kovProk = Array("", "")
    v = Лист1.Cells(row + 1, s1)
    If v <> "" Then
        a = Split(v, " ")
        get_kovProk = Array(Лист1.Cells(1, s1), a(1))
    End If
End Function

Sub SaveFile()
    Application.ScreenUpdating = False
    dd = Day(Лист2.Cells(1, 11))
    If dd < 10 Then dd = "0" & dd
    mm = Month(Now)
    If mm < 10 Then mm = "0" & mm
    hh = Hour(Now)
    If hh < 10 Then hh = "0" & hh
    mmm = Minute(Now)
    If mmm < 10 Then mmm = "0" & mmm
    fname = Year(Now) & "-" & mm & "-" & dd & " " & Environ("USERNAME")

    On Error Resume Next
    MkDir (ThisWorkbook.Path & "\Архив")
    MkDir (ThisWorkbook.Path & "\Архив\" & Year(Now) & "-" & mm)

    ActiveWorkbook.SaveCopyAs Filename:=ThisWorkbook.Path & "\Архив\" & Year(Now) & "-" & mm & "\" & fname & ".xlsm"
    Application.ScreenUpdating = True
End Sub