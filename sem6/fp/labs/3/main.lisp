(defun calc-sum(mat)
	(let*
		(
			(m (array-dimension mat 0))
			(n (array-dimension mat 1))
			(rows (make-array (list m) :initial-element 0))
			(cols (make-array (list n) :initial-element 0))
			(res (make-array (list m n)))
			(total 0)
		)
 
		(loop for i from 0 to (- m 1) do
			(loop for j from 0 to (- n 1) do
				(setf (aref rows i) (+ (aref rows i) (aref mat i j)))
				(setq total (+ total (aref mat i j)))
			)
		)
 
		(loop for j from 0 to (- n 1) do
			(loop for i from 0 to (- m 1) do
				(setf (aref cols j) (+ (aref cols j) (aref mat i j)))
			)
		)
 
		(loop for i from 0 to (- m 1) do
			(loop for j from 0 to (- n 1) do
				(setf (aref res i j) (- total (- (+ (aref rows i) (aref cols j)) (aref mat i j))))
			)
		)
 
		res
	)
)
 
(print (calc-sum (make-array '(3 2) :initial-contents '((1 2) (3 4) (5 6)))))
(print (calc-sum (make-array '(2 2) :initial-contents '((4 8) (1 7)))))
(print (calc-sum (make-array '(1 1) :initial-contents '((42)))))
