global bubbleSortAsm

section .text

bubbleSortAsm:
	push ebp
	mov ebp, esp
	
	mov edx, [ebp + 12]
	mov esi, [ebp + 8]

	cmp edx, 2
	jl end

	sub edx, 1

loop1:
	mov ecx, 0

loop2:
	lea edi, [esi + ecx * 4]
	
	mov eax, [edi]
	mov ebx, [edi + 4]

	cmp eax, ebx
	jle skip

	push eax
	push ebx
	pop dword [edi]
	pop dword [edi + 4]

skip:
	add ecx, 1
	cmp ecx, edx
	jl loop2

	sub edx, 1
	cmp edx, 0
	jg loop1
	
end:
	mov esp, ebp
	pop ebp
	
	ret
