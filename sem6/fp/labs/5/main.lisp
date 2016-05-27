(defclass cart()
	(
		(x :initarg :x :reader cart-x)
		(y :initarg :y :reader cart-y)
	)
)

(defmethod print-object ((c cart) stream)
	(format stream "[Cart x ~d y ~d]"
		(cart-x c) (cart-y c)
	)
)

(defclass line()
	(
		(start :initarg :start :accessor line-start)
		(end   :initarg :end   :accessor line-end)
	)
)

(defmethod print-object((lin line) stream)
	(format stream "[Line ~s ~s]"
		(line-start lin) (line-end lin)
	)
)

(defmethod sub2((c1 cart) (c2 cart))
	(make-instance 'cart
		:x (- (cart-x c1) (cart-x c2))
		:y (- (cart-y c1) (cart-y c2))
	)
)

(defun sqr(num)
	(* num num)
)

(defmethod vec-length((c cart))
	(sqrt (+ (sqr (cart-x c)) (sqr (cart-y c))))
)

(defun line-length(lin)
	(vec-length (sub2 (line-start lin) (line-end lin)))
)

;; Пример 1
(setq lin (make-instance 'line
	:start (make-instance 'cart :x 4 :y 3)
	:end (make-instance 'cart :x 0 :y 0))
)

(print (line-length lin))

;; Пример 2
(setq lin (make-instance 'line
	:start (make-instance 'cart :x 4 :y 3)
	:end (make-instance 'cart :x 4 :y 5))
)

(print (line-length lin))
