(defun count-words-starting-with-char(ch str)
	(let
		(
			(res 0)
			(cur-ch nil)
			(word? T)
		)
		
		(loop for i from 0 to (- (length str) 1) do
			(setq cur-ch (char str i))
			
			(if (alpha-char-p cur-ch)
				(progn
					(if (and word? (char= ch cur-ch))
						(setq res (+ res 1))
					)
					
					(setq word? nil)
				)
				
				(setq word? T)
			)
		)
		
		res
	)
)

(print (count-words-starting-with-char #\t "This is test"))
(print (count-words-starting-with-char #\f "Another string for testing"))
(print (count-words-starting-with-char #\a "A lot of aaa aaaa aaaaaaaa a"))
(print (count-words-starting-with-char #\z "Zero?"))
(print (count-words-starting-with-char #\L "Last"))
