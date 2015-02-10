; xxxFib заменить xxx СВОИМ персональным кодом
(define(baaFib n)
   (fib-iter 1 0 n))
(define(fib-iter a b n)
  (if(= n 0)
     b
     (fib-iter (+ a b)a(- n 1))))
"baaFib(047)="
(baaFib 047)

;заменить N СВОИМ номером по списку группы

"baaFib(5)="
(baaFib 5)
