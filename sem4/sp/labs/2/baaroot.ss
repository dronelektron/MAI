;xxxroot
(define (fixed-point first-guess)
  (try first-guess))
(define(try guess)
  (define next (newton-transform guess))
  (display "+")
  (if (close-enough? guess next)
      next
      (try next)))
(define (close-enough? x y)
  (<(abs (- x y))tolerance))
(define(newton-transform x)
  (- x (/(fun x)(deriv x))))
(define(deriv x)(/(-(fun (+ x dx))
                    (fun x))
                  dx))
(define(fun x)
  (define z (- x (/ 101 102)))
;=========================================================
;C++: 2*exp(-z) + sin(z+pi) + tan(z) + expt(z-1,2) + -0.17
;=========================================================
  ;e^z + ln(z) - 10z = 0, z = x - 101 / 102; 4
  (- (+ (exp z) (log z)) (* 10 z))
  ;(+(* 2 (exp (- z)))
  ;  (sin (+ z pi))
  ;  (tan z)
  ;  (expt(- z 1)2)
  ;  -0.17)
)

(define dx 1e-5)
(define tolerance 1e-5)
(define (root first-gess)
  (define temp(fixed-point first-gess))
  (newline)
  (display"first-gess=\t")
  (display first-gess)(newline)
  (display"discrepancy=\t")
  (display(fun temp))(newline)
  (display"root=\t\t")
  temp 
)

"baa variant 1"
(root 4)
;3.271