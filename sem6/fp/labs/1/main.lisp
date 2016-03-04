(defun f-iter(n3 n2 n1 count)
	(if (= count 0)
		n1
		(f-iter (+ n3 n2 n1) n3 n2 (- count 1))))

(defun f(n)
	(f-iter 2 1 0 n))

(loop for x from 1 to 50 do
	(format t "f(~d) = ~d~%" x (f x)))
