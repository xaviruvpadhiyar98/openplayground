async function stream() {
    var content = ''
    const resultDiv = document.getElementById('content');
    var content = content + document.getElementById('content').innerHTML.split("<")[0].split("\n")[0]

    var combinedText = Array.from(document.querySelectorAll('span[name="contentText"]'))
    .reduce((acc, span) => acc + span.innerHTML, '');


    var content = content + combinedText

    data = {
        model: document.getElementById('models').value,
        content: content,
        max_new_tokens: parseFloat(document.getElementById('maximumLength').value),
        temperature: parseFloat(document.getElementById('temperature').value, 0.6),
        top_p: parseFloat(document.getElementById('topP').value),
        top_k: parseInt(document.getElementById('topK').value, 10),
        repetition_penalty: parseFloat(document.getElementById('repetitionPenalty').value)
    }

    const response = await fetch('/stream', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        },
    });

    const reader = response.body.getReader();

    while (true) {
        const { done, value } = await reader.read();

        if (done) {
            break;
        }

        // Convert the Uint8Array to a string and append it to the div
        const text = new TextDecoder().decode(value);
        resultDiv.innerHTML += text
    }
}


async function reset() {
    document.getElementById('content').innerHTML = '<span name="contentText">Once upon a time,</span>';
}